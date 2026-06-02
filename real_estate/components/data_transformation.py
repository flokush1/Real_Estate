import os
import re
import sys
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

from real_estate.entity import (
    DataTransformationConfig,
    DataTransformationArtifact,
    DataIngestionArtifact,
)
from real_estate.exception.exception import RealEstateException
from real_estate.logging.logger import logging
from real_estate.constant import TARGET_COLUMN, CITY_LOCALITIES_JSON, CIRCLE_RATES_DIR
from real_estate.utils.description_parser import (
    combine_text,
    extract_age_of_property,
    extract_area_sqft,
    extract_area_value_and_unit,
    extract_balconies,
    extract_bathrooms,
    extract_bhk,
    extract_facing,
    extract_furnishing,
    extract_price,
    extract_road_width_ft,
    has_garden_park,
    has_main_road,
    has_parking,
    has_pool,
    is_corner,
    is_gated,
)
from sklearn.neighbors import BallTree
from real_estate.utils.locality_matcher import LocalityMatcher
from real_estate.utils.circle_rate_matcher import CircleRateMatcher, _resolve_city


# Maximum distance allowed when borrowing a neighbour's circle rate.
# Rows whose nearest donor is farther than this will remain NaN and be dropped.
MAX_CIRCLE_RATE_FALLBACK_KM: float = 5.0

# ────────────────────────────────────────────────────────────────
#  AREA CONVERSION FACTORS  →  everything to Sq-ft
# ────────────────────────────────────────────────────────────────
_SQFT_CONVERSION = {
    # ── square feet variants ──────────────────────────────────
    "sq-ft": 1.0, "sq.ft": 1.0, "sqft": 1.0, "sq ft": 1.0,
    "sq. ft": 1.0, "sq. ft.": 1.0, "sq.ft.": 1.0, "sqft.": 1.0,
    "sft": 1.0, "square feet": 1.0, "square ft": 1.0, "square foot": 1.0,
    # ── square yard variants ──────────────────────────────────
    "sq-yrd": 9.0, "sq.yd": 9.0, "sqyd": 9.0, "sq yd": 9.0,
    "sq. yd": 9.0, "sq. yd.": 9.0, "sq.yd.": 9.0, "sqyrd": 9.0,
    "sqyrd.": 9.0, "sq yds": 9.0, "sq. yds": 9.0, "sq. yds.": 9.0,
    "sq yards": 9.0, "sq. yards": 9.0, "sq. yards.": 9.0, "sq.yards.": 9.0,
    "sq-yard": 9.0, "sq yard": 9.0, "sqyds": 9.0,
    "square yard": 9.0, "square yards": 9.0, "square yd": 9.0,
    # ── square metre variants ─────────────────────────────────
    "sq-m": 10.7639, "sq.m": 10.7639, "sqm": 10.7639, "sq m": 10.7639,
    "sq.m.": 10.7639, "sq. m": 10.7639, "sq. m.": 10.7639, "sqm.": 10.7639,
    "sq.mt": 10.7639, "sq. mt": 10.7639, "sq. mt.": 10.7639, "sqmt": 10.7639,
    "sq mtr": 10.7639, "sq. mtr": 10.7639, "sq. mtr.": 10.7639, "sqmtr": 10.7639,
    "sq meter": 10.7639, "sq-meter": 10.7639, "sq. meter": 10.7639,
    "sq. meters": 10.7639, "sq metre": 10.7639, "sq. metre": 10.7639,
    "square meter": 10.7639, "square meters": 10.7639,
    "square metre": 10.7639, "square metres": 10.7639,
    # ── others ────────────────────────────────────────────────
    "acre": 43560.0,
    "bigha": 27000.0,
    "hectare": 107639.0,
    "marla": 272.25,
    "ground": 2400.0,
    "rood": 10890.0,
    "biswa": 1350.0, "biswa1": 1350.0, "biswa2": 1350.0,
    "kanal": 5445.0,                    # 1 Kanal = 20 Marla = 5445 sq ft (Punjab/Haryana)
    "aankadam": 72.0,                   # 1 Aankadam = 72 sq ft (Tamil Nadu)
    "gaj": 9.0,                         # 1 Gaj = 1 sq yard = 9 sq ft
    "are": 1076.39,                     # 1 Are = 100 sq metres = 1076.39 sq ft
    "chatak": 45.0,                     # 1 Chatak = 45 sq ft (Bengal/Eastern India)
    "guntha": 1089.0,                   # 1 Guntha = 1/40 acre = 1089 sq ft (Maharashtra/Goa)
    "gunta": 1089.0,                    # alternate spelling
    "guntha.": 1089.0,
    "gunta.": 1089.0,
}


def _normalize_city_name(value: str) -> str:
    value = str(value or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


_EXCLUDED_CITY_NAMES = {
    _normalize_city_name(name)
    for name in [
        "Asagarpur Jagir",
        "Aurangabad",
        "Baghpat",
        "Bajidpur",
        "Bela Kalan",
        "Bharatpur",
        "Bhiwadi",
        "Bisokhar",
        "Bulandshahr",
        "Dadri",
        "Desai Village",
        "Dharuhera",
        "Dungarpur Rilka",
        "Eachachhar",
        "Garhi Bohar",
        "Gulistanpur",
        "Hamjapur",
        "Hapur",
        "Hodal Rural",
        "Jhanjhari",
        "Jharli",
        "Jhatta",
        "Kalyanpura",
        "Khera Choganpur",
        "Kulesara",
        "Malpura",
        "Mewat",
        "Mishripur",
        "Nangla Rudh",
        "Nangli Umarpur",
        "Nasirpur",
        "Neemrana",
        "Pabhi Sadakpur",
        "Patli Khurd",
        "Rampur Jagir",
        "Rundh Bhakhera",
        "Sahapur Khurd",
        "Sankhol",
        "Sarai Ahmed",
        "Sarhol",
        "Shahjahanpur",
        "Shahpur",
        "Shahpur Govardhanpur Khadar",
        "Shamli",
        "Tilhar",
        "Uchana",
        "Ward No 8",
        "Yusufpur Chak Saberi",
    ]
}


class DataTransformation:
    """
    Full feature-engineering pipeline on the merged dataset.

    Steps:
        1. Basic clean (dedup, drop null target)
        2. Fill missing values from description / amenities
        3. age_of_property → standardise + bucket dummies
        4. Convert covered_area to sqft
        5. Furnishing type dummies
        6. price_per_sqft column
        7. Boolean features (parking, pool, main_road, garden_park)
        8. Facing direction standardise
        9. Save transformed CSV
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        config: DataTransformationConfig = DataTransformationConfig(),
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.config = config
        
        # Initialize locality matcher from per-city JSON
        try:
            self.locality_matcher = LocalityMatcher(CITY_LOCALITIES_JSON)
            logging.info(f"Loaded {self.locality_matcher.total_localities} localities across {len(self.locality_matcher._city_choices)} cities")
        except Exception as e:
            logging.warning(f"Could not load city localities JSON: {e}. Locality filling will be skipped.")
            self.locality_matcher = None

        # Initialize circle-rate matcher
        try:
            self.circle_rate_matcher = CircleRateMatcher(CIRCLE_RATES_DIR)
            logging.info(f"Loaded {self.circle_rate_matcher.total_entries} circle-rate entries")
        except Exception as e:
            logging.warning(f"Could not load circle rates: {e}. Circle rate column will be NaN.")
            self.circle_rate_matcher = None

    @staticmethod
    def _coerce_numeric_columns(
        df: pd.DataFrame,
        columns: list[str],
        context: str,
    ) -> pd.DataFrame:
        """
        Force known numeric columns to numeric dtype and null out invalid text.
        """
        for column in columns:
            if column not in df.columns:
                continue

            before_non_null = df[column].notna().sum()
            df[column] = pd.to_numeric(df[column], errors="coerce")
            dropped_values = before_non_null - df[column].notna().sum()
            if dropped_values > 0:
                logging.info(
                    f"  {context}: coerced {column} to numeric and nulled {dropped_values} invalid values"
                )

        return df

    # ============================================================
    #  STEP 0b – drop unwanted columns
    # ============================================================
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 0b: Drop unwanted columns")
        cols_to_drop = ["backlane", "rectangular_plot", "source"]
        existing = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing)
        logging.info(f"  Dropped columns: {existing}")
        return df

    # ============================================================
    #  STEP 1 – basic clean
    # ============================================================
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 1: Basic cleaning")
        before = len(df)
        df = df.drop_duplicates()
        logging.info(f"  Dropped duplicates: {before} -> {len(df)}")

        if TARGET_COLUMN in df.columns:
            before = len(df)
            df = df.dropna(subset=[TARGET_COLUMN])
            logging.info(f"  Dropped null target rows: {before} -> {len(df)}")

        # Drop rows with missing latitude or longitude
        lat_col  = "latitude"  if "latitude"  in df.columns else None
        lon_col  = "longitude" if "longitude" in df.columns else None
        latlon_cols = [c for c in [lat_col, lon_col] if c]
        if latlon_cols:
            before = len(df)
            df = df.dropna(subset=latlon_cols)
            logging.info(f"  Dropped rows with null lat/lon: {before} -> {len(df)} (removed {before - len(df)})")
        else:
            logging.warning("  latitude/longitude columns not found, skipping lat-lon filter")

        return df

    # ============================================================
    #  STEP 1b – remove user-excluded city names
    # ============================================================
    def _remove_excluded_cities(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 1b: Remove excluded city names")

        if "city" not in df.columns:
            logging.warning("  city column not found, skipping excluded-city filter")
            return df

        city_norm = df["city"].fillna("").astype(str).map(_normalize_city_name)
        remove_mask = city_norm.isin(_EXCLUDED_CITY_NAMES)

        removed = int(remove_mask.sum())
        before = len(df)
        if removed > 0:
            df = df.loc[~remove_mask].copy()

        logging.info(f"  Excluded-city filter: {before} -> {len(df)} (removed {removed})")
        return df

    # ============================================================
    #  STEP 2 – extract values from description & replace originals
    # ============================================================
    def _fill_from_description(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 2: Extract values from description/amenities & replace originals")
        df = self._coerce_numeric_columns(
            df,
            ["bhk", "bathrooms", "balconies", "price_numeric"],
            "pre-description fill",
        )
        text = df.apply(combine_text, axis=1)

        # --- Create desc_* columns with extracted values ---
        logging.info("  Creating desc_* columns with extracted values...")
        
        df["desc_bhk"] = text.apply(extract_bhk)
        df["desc_bathrooms"] = text.apply(extract_bathrooms)
        df["desc_balconies"] = text.apply(extract_balconies)
        df["desc_price"] = text.apply(extract_price)
        
        # Extract area value and unit
        area_data = text.apply(extract_area_value_and_unit)
        df["desc_area"] = area_data.apply(lambda x: x[0])
        df["desc_unit"] = area_data.apply(lambda x: x[1])
        
        logging.info(f"  desc_bhk: extracted {df['desc_bhk'].notna().sum()} values")
        logging.info(f"  desc_bathrooms: extracted {df['desc_bathrooms'].notna().sum()} values")
        logging.info(f"  desc_balconies: extracted {df['desc_balconies'].notna().sum()} values")
        logging.info(f"  desc_price: extracted {df['desc_price'].notna().sum()} values")
        logging.info(f"  desc_area: extracted {df['desc_area'].notna().sum()} values")
        logging.info(f"  desc_unit: extracted {df['desc_unit'].notna().sum()} values")
        
        # --- Convert desc_area to sqft using desc_unit ---
        logging.info("  Converting desc_area to square feet...")
        
        def convert_desc_area_to_sqft(row):
            val = row["desc_area"]
            unit = row["desc_unit"]
            if pd.isna(val) or pd.isna(unit):
                return val
            factor = _SQFT_CONVERSION.get(str(unit).lower().strip(), None)
            if factor is None:
                logging.warning(f"  Unknown desc_unit '{unit}', keeping value as-is")
                return val
            return val * factor
        
        df["desc_area_sqft"] = df.apply(convert_desc_area_to_sqft, axis=1)
        logging.info(f"  desc_area_sqft: converted {df['desc_area_sqft'].notna().sum()} values")
        
        # --- Replace original columns with desc_* (use original if desc_* is missing) ---
        logging.info("  Replacing original columns with desc_* values...")
        
        # Replace logic: use desc_* if available, otherwise keep original
        original_bhk_na = df["bhk"].isna().sum()
        df["bhk"] = df["desc_bhk"].fillna(df["bhk"])
        logging.info(f"    bhk: replaced with desc_bhk (filled {df['bhk'].notna().sum() - (len(df) - original_bhk_na)} additional)")
        
        original_bathrooms_na = df["bathrooms"].isna().sum()
        df["bathrooms"] = df["desc_bathrooms"].fillna(df["bathrooms"])
        logging.info(f"    bathrooms: replaced with desc_bathrooms (filled {df['bathrooms'].notna().sum() - (len(df) - original_bathrooms_na)} additional)")
        
        original_balconies_na = df["balconies"].isna().sum()
        df["balconies"] = df["desc_balconies"].fillna(df["balconies"])
        logging.info(f"    balconies: replaced with desc_balconies (filled {df['balconies'].notna().sum() - (len(df) - original_balconies_na)} additional)")

        df = self._coerce_numeric_columns(
            df,
            ["bhk", "bathrooms", "balconies"],
            "post-description fill",
        )

        # ── Bounds sanity: null out unreasonable values ──────────
        for col, cap in [("bhk", 12), ("bathrooms", 12), ("balconies", 12)]:
            bad = df[col].notna() & (df[col] > cap)
            if bad.any():
                logging.info(f"    {col}: nulled {bad.sum()} rows with value > {cap}")
                df.loc[bad, col] = np.nan
        
        original_price_na = df["price_numeric"].isna().sum()
        df["price_numeric"] = df["desc_price"].fillna(df["price_numeric"])
        df = self._coerce_numeric_columns(df, ["price_numeric"], "post-price fill")
        logging.info(f"    price_numeric: replaced with desc_price (filled {df['price_numeric'].notna().sum() - (len(df) - original_price_na)} additional)")
        
        # For area: keep desc_area_sqft as a separate column for manual comparison.
        # Do NOT overwrite covered_area_value with desc_area_sqft.
        logging.info(f"    covered_area_value: kept original ({df['covered_area_value'].notna().sum()} non-null)")
        logging.info(f"    desc_area_sqft: available for comparison ({df['desc_area_sqft'].notna().sum()} non-null)")
        
        # --- Fill other fields from description (not creating desc_* for these) ---
        # facing_direction
        mask = df["facing_direction"].isna()
        if mask.any():
            df.loc[mask, "facing_direction"] = text[mask].apply(extract_facing)
            logging.info(f"    facing_direction: filled {mask.sum() - df['facing_direction'].isna().sum()} of {mask.sum()} missing")
        
        # age_of_property
        mask = df["age_of_property"].isna()
        if mask.any():
            df.loc[mask, "age_of_property"] = text[mask].apply(extract_age_of_property)
            logging.info(f"    age_of_property: filled {mask.sum() - df['age_of_property'].isna().sum()} of {mask.sum()} missing")
        
        # furnishing_type
        mask = df["furnishing_type"].isna()
        if mask.any():
            df.loc[mask, "furnishing_type"] = text[mask].apply(extract_furnishing)
            logging.info(f"    furnishing_type: filled {mask.sum() - df['furnishing_type'].isna().sum()} of {mask.sum()} missing")
        
        return df

    # ============================================================
    #  STEP 3 – age_of_property → standardise + bucket dummies
    # ============================================================
    def _transform_age(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 3: age_of_property -> standardise (no OHE dummies)")

        def _standardise_age(val):
            if not isinstance(val, str):
                return np.nan
            val_lower = val.lower().strip()
            if "new" in val_lower:
                return "New Construction"
            m = re.search(r"(\d+)\s*year", val_lower)
            if m:
                age = int(m.group(1))
                if age < 5:
                    return "Less than 5 years"
                elif age <= 10:
                    return "5 to 10 years"
                elif age <= 20:
                    return "10 to 20 years"
                else:
                    return "Above 20 years"
            if "less than 5" in val_lower:
                return "Less than 5 years"
            if "5 to 10" in val_lower:
                return "5 to 10 years"
            if "10 to 15" in val_lower or "10 to 20" in val_lower or "15 to 20" in val_lower:
                return "10 to 20 years"
            if "above 20" in val_lower or "20+" in val_lower:
                return "Above 20 years"
            return np.nan

        df["age_of_property"] = df["age_of_property"].apply(_standardise_age)
        logging.info(f"  age_of_property standardised. value counts:\n{df['age_of_property'].value_counts(dropna=False).to_string()}")
        return df

    # ============================================================
    #  STEP 4 – convert all area units to sqft
    # ============================================================
    def _convert_area_to_sqft(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 4: Convert covered_area to sqft")

        def _to_sqft(row):
            val = row["covered_area_value"]
            unit = row["covered_area_unit"]
            if pd.isna(val) or pd.isna(unit):
                return val
            unit_key = str(unit).lower().strip().rstrip(".")
            factor = _SQFT_CONVERSION.get(unit_key, None)
            if factor is None:
                # try with trailing dot preserved
                factor = _SQFT_CONVERSION.get(str(unit).lower().strip(), None)
            if factor is None:
                logging.warning(f"  Unknown unit '{unit}', keeping as-is")
                return val
            return val * factor

        df["covered_area_sqft"] = df.apply(_to_sqft, axis=1)

        # Null out area < 225 sqft (unreasonably small)
        bad_area = df["covered_area_sqft"].notna() & (df["covered_area_sqft"] < 225)
        if bad_area.any():
            logging.info(f"  Nulled {bad_area.sum()} rows with covered_area_sqft < 225")
            df.loc[bad_area, "covered_area_sqft"] = np.nan

        df["covered_area_unit"] = df["covered_area_unit"].where(
            df["covered_area_sqft"].isna(), "Sq-ft"
        )
        logging.info(f"  covered_area_sqft nulls: {df['covered_area_sqft'].isna().sum()}")
        return df

    # ============================================================
    #  STEP 5 – furnishing type → 3-category standardise (no OHE)
    # ============================================================
    def _furnishing_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 5: Furnishing type -> 3-category standardise")

        furn_map = {
            "Furnished":       "Furnished",
            "Fully Furnished": "Furnished",
            "Semi-Furnished":  "Semi-Furnished",
            "Semi Furnished":  "Semi-Furnished",
            "Unfurnished":     "Unfurnished",
        }
        df["furnishing_type"] = df["furnishing_type"].map(furn_map)
        logging.info(f"  furnishing_type value counts:\n{df['furnishing_type'].value_counts(dropna=False).to_string()}")
        return df

    # ============================================================
    #  STEP 5b – property_type → consolidated OHE
    # ============================================================
    def _property_type_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 5b: property_type -> keep 5 types, drop others, OHE")

        # ── Consolidation map ────────────────────────────────────
        _CONSOLIDATION = {
            "Apartment":               "apartment",
            "Studio Apartment":        "apartment",
            "Studio":                  "apartment",
            "Builder Floor Apartment": "builder_floor",
            "Builder Floor":           "builder_floor",
            "Independent Floor":       "builder_floor",
            "Independent House":       "res_house",
            "Residential House":       "res_house",
            "Farm House":              "res_house",
            "Plot":                    "plot",
            "Residential Plot":        "plot",
            "Villa":                   "villa",
        }

        _KEEP = {"apartment", "builder_floor", "villa", "plot", "res_house"}

        df["property_type_grouped"] = df["property_type"].map(_CONSOLIDATION).fillna(
            df["property_type"].str.lower().str.replace(" ", "_", regex=False)
        )

        # Drop rows whose property type is not in the 5 kept categories
        before = len(df)
        df = df[df["property_type_grouped"].isin(_KEEP)].copy()
        logging.info(f"  Dropped {before - len(df)} rows with non-target property types. Remaining: {len(df)}")
        logging.info(f"  property_type_grouped value counts:\n{df['property_type_grouped'].value_counts().to_string()}")

        dummies = pd.get_dummies(df["property_type_grouped"], prefix="prop_type", dtype=int)
        df = pd.concat([df, dummies], axis=1)
        logging.info(f"  prop_type OHE columns: {list(dummies.columns)}")
        return df

    # ============================================================
    #  STEP 6 – price_per_sqft
    # ============================================================
    def _add_price_per_sqft(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 6: Create price_per_sqft column")
        df["price_per_sqft"] = df["price_numeric"] / df["covered_area_sqft"].replace(0, np.nan)
        before = len(df)
        df = df.dropna(subset=["price_per_sqft"]).copy()
        logging.info(f"  Dropped {before - len(df)} rows with null price_per_sqft. Remaining: {len(df)}")
        return df

    # ============================================================
    #  STEP 7 – boolean features from description + amenities
    # ============================================================
    def _add_boolean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 7: Boolean features from description/amenities")
        text = df.apply(combine_text, axis=1)

        df["is_parking"] = text.apply(has_parking).astype(int)
        df["is_pool"] = text.apply(has_pool).astype(int)
        df["is_main_road"] = text.apply(has_main_road).astype(int)
        df["is_garden_park"] = text.apply(has_garden_park).astype(int)
        df["is_gated"] = text.apply(is_gated).astype(int)
        df["is_corner"] = text.apply(is_corner).astype(int)
        df["road_width_ft"] = text.apply(extract_road_width_ft)

        for col in ["is_parking", "is_pool", "is_main_road", "is_garden_park",
                    "is_gated", "is_corner"]:
            logging.info(f"  {col}: {df[col].sum()} / {len(df)} = {df[col].mean():.2%}")
        non_null = df["road_width_ft"].notna().sum()
        logging.info(f"  road_width_ft: {non_null} non-null / {len(df)} = {non_null/len(df):.2%}")
        return df

    # ============================================================
    #  STEP 8 – facing direction standardise
    # ============================================================
    def _standardise_facing(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 8: Standardise facing_direction")
        face_map = {
            "East": "East",
            "East facing": "East",
            "West": "West",
            "North": "North",
            "North facing": "North",
            "South": "South",
            "South facing": "South",
            "North - East": "North-East",
            "North-East facing": "North-East",
            "North - West": "North-West",
            "South - East": "South-East",
            "South -West": "South-West",
            "South-West facing": "South-West",
        }
        df["facing_direction"] = df["facing_direction"].map(face_map)
        logging.info(f"  facing_direction nulls after standardise: {df['facing_direction'].isna().sum()}")
        return df

    # ============================================================
    #  STEP 8b – possession_status → Ready to Move / Under Construction
    # ============================================================
    def _standardise_possession(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 8b: Standardise possession_status")

        if "possession_status" not in df.columns:
            logging.warning("  possession_status column not found, skipping")
            return df

        today = datetime.now()
        cutoff = today + relativedelta(months=6)

        # Ordinal suffixes for day parsing: "29th", "1st", "23rd", "2nd"
        _ORD_RE = re.compile(r"(\d+)(?:st|nd|rd|th)")

        def _parse_possession(val):
            if not isinstance(val, str) or not val.strip():
                return np.nan

            v = val.strip()
            vl = v.lower()

            # ── keyword matches ──────────────────────────────
            if vl in ("ready to move", "ready to move in",
                      "immediate", "immediately", "completed"):
                return "Ready to Move"
            if vl == "under construction":
                return "Under Construction"

            # ── try full date: "29th Jan, 2026" ──────────────
            clean = _ORD_RE.sub(r"\1", v)          # strip ordinal suffix
            clean = clean.replace(",", "").strip()  # drop comma
            for fmt in ("%d %b %Y", "%d %B %Y", "%d %b%Y"):
                try:
                    dt = datetime.strptime(clean, fmt)
                    return "Ready to Move" if dt <= cutoff else "Under Construction"
                except ValueError:
                    continue

            # ── abbreviated: "Nov '30" → Nov 2030 ────────────
            m = re.match(r"([A-Za-z]+)\s*['’](\d{2})$", v)
            if m:
                try:
                    month_str = m.group(1)
                    year_2d = int(m.group(2))
                    year = 2000 + year_2d
                    dt = datetime.strptime(f"1 {month_str} {year}", "%d %b %Y")
                    return "Ready to Move" if dt <= cutoff else "Under Construction"
                except ValueError:
                    pass

            return np.nan

        df["possession_status"] = df["possession_status"].apply(_parse_possession)
        logging.info(f"  possession_status value counts:\n{df['possession_status'].value_counts(dropna=False).to_string()}")
        return df
    
    # ============================================================
    #  STEP 9 – fill locality from description and address
    # ============================================================
    def _fill_locality(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 9: Fill locality from description and address")

        if self.locality_matcher is None:
            logging.warning("  Locality matcher not available, skipping locality filling")
            return df

        if 'locality' not in df.columns:
            logging.warning("  No locality column found, skipping")
            return df

        before_missing = df['locality'].isna().sum()
        logging.info(f"  Localities missing before: {before_missing}")

        # Snapshot original locality before filling
        locality_original = df['locality'].copy()
        locality_clean = df['locality'].fillna('').astype(str).str.strip()
        missing_mask = locality_clean.eq('') | locality_clean.str.lower().eq('nan')
        sector_enrichment_mask = (~missing_mask) & ~locality_clean.str.contains('sector', case=False, na=False)

        logging.info(f"  Rows requiring locality match: {int(missing_mask.sum())}")
        logging.info(f"  Rows eligible for sector enrichment: {int(sector_enrichment_mask.sum())}")

        # Apply city-aware locality extraction
        def extract_locality_row(row):
            return self.locality_matcher.extract_locality(
                city=row.get('city', ''),
                description=row.get('description', ''),
                address=row.get('address_full', ''),
                current_locality=row.get('locality', ''),
            )

        def enrich_locality_with_sector(row):
            locality = row.get('locality', '')
            locality = locality.strip() if isinstance(locality, str) else ''
            if not locality:
                return locality

            sector = self.locality_matcher._extract_sector(row.get('address_full', ''))
            if sector is None:
                sector = self.locality_matcher._extract_sector(row.get('description', ''))

            if sector and 'sector' not in locality.lower():
                return f"{locality}, {sector}"
            return locality

        if missing_mask.any():
            df.loc[missing_mask, 'locality'] = df.loc[missing_mask].apply(extract_locality_row, axis=1)

        if sector_enrichment_mask.any():
            df.loc[sector_enrichment_mask, 'locality'] = df.loc[sector_enrichment_mask].apply(
                enrich_locality_with_sector,
                axis=1,
            )

        after_missing = df['locality'].isna().sum()
        filled = before_missing - after_missing
        logging.info(f"  Localities filled: {filled}")
        logging.info(f"  Localities still missing: {after_missing}")

        # ── Save locality_filled.csv for review ────────────────
        try:
            def _tag_source(orig, filled_val):
                if isinstance(orig, str) and orig.strip():
                    return "original"
                if isinstance(filled_val, str) and filled_val.strip():
                    return "filled"
                return "still_missing"

            slim = pd.DataFrame({
                "city":              df.get("city", ""),
                "address_full":      df.get("address_full", ""),
                "description":       df.get("description", ""),
                "locality_original": locality_original,
                "locality_filled":   df["locality"],
                "source_of_fill":    [
                    _tag_source(o, f)
                    for o, f in zip(locality_original, df["locality"])
                ],
            })

            os.makedirs(self.config.transformed_data_dir, exist_ok=True)
            locality_csv_path = os.path.join(
                self.config.transformed_data_dir, "locality_filled.csv"
            )
            slim.to_csv(locality_csv_path, index=False)
            newly = (slim["source_of_fill"] == "filled").sum()
            still = (slim["source_of_fill"] == "still_missing").sum()
            logging.info(
                f"  locality_filled.csv saved -> {locality_csv_path}  "
                f"(newly_filled={newly}, still_missing={still})"
            )
        except Exception as e:
            logging.warning(f"  Could not save locality_filled.csv: {e}")

        return df

    # ============================================================
    #  STEP 10 – circle rate from (city, locality)
    # ============================================================

    @staticmethod
    def _map_prop_type_for_circle(property_type: str) -> str:
        val = str(property_type or "").strip().lower()
        if "commercial" in val:
            return "Commercial"
        if "agricultural" in val:
            return "Agricultural"
        if "institution" in val:
            return "Institutional"
        return "Residential"

    @staticmethod
    def _normalize_city_key(value) -> str:
        if pd.isna(value):
            return ""
        resolved = _resolve_city(str(value))
        if resolved is not None:
            return resolved
        return re.sub(r"\s+", " ", str(value).strip().lower())

    def _fill_circle_rate_by_nearest_haversine(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing circle_rate values using the nearest known row by lat/lon
        (haversine distance via sklearn BallTree).

        Donor pool priority:
          1) Same city + same mapped property type
          2) Same city (any property type)
          3) Same mapped property type (any city)
          4) Any row with a known circle_rate

        This is purely spatial — no default rates are hardcoded, so new cities
        automatically benefit as soon as any rows from that city have a known rate.
        """
        if "circle_rate" not in df.columns:
            return df

        required_cols = ["latitude", "longitude"]
        if any(col not in df.columns for col in required_cols):
            return df

        df = df.copy()
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["circle_rate"] = pd.to_numeric(df["circle_rate"], errors="coerce")

        if "property_type" in df.columns:
            df["_circle_prop_type"] = df["property_type"].apply(self._map_prop_type_for_circle)
        else:
            df["_circle_prop_type"] = "Residential"

        if "city" in df.columns:
            df["_circle_city_key"] = df["city"].apply(self._normalize_city_key)
        else:
            df["_circle_city_key"] = "all"

        valid_coord_mask = df["latitude"].notna() & df["longitude"].notna()
        known_mask = valid_coord_mask & df["circle_rate"].notna()
        missing_mask = valid_coord_mask & df["circle_rate"].isna()

        if known_mask.sum() == 0 or missing_mask.sum() == 0:
            return df.drop(columns=["_circle_prop_type", "_circle_city_key"], errors="ignore")

        earth_radius_km = 6371.0088
        filled_rows = 0

        group_cols = ["_circle_city_key", "_circle_prop_type"]
        missing_groups = df.loc[missing_mask].groupby(group_cols, dropna=False).groups

        for (city_key, prop_type), missing_index in missing_groups.items():
            donor_mask = (
                known_mask
                & (df["_circle_city_key"] == city_key)
                & (df["_circle_prop_type"] == prop_type)
            )
            if donor_mask.sum() == 0:
                donor_mask = known_mask & (df["_circle_city_key"] == city_key)
            if donor_mask.sum() == 0:
                donor_mask = known_mask & (df["_circle_prop_type"] == prop_type)
            if donor_mask.sum() == 0:
                donor_mask = known_mask

            donor_index = df.index[donor_mask]
            if donor_index.empty:
                continue

            target_index = pd.Index(missing_index)
            donor_coords = np.radians(
                df.loc[donor_index, ["latitude", "longitude"]].to_numpy(dtype=float)
            )
            target_coords = np.radians(
                df.loc[target_index, ["latitude", "longitude"]].to_numpy(dtype=float)
            )

            tree = BallTree(donor_coords, metric="haversine")
            dist_rad, nearest_pos = tree.query(target_coords, k=1)

            nearest_index = donor_index.to_numpy()[nearest_pos[:, 0]]
            dist_km = dist_rad[:, 0] * earth_radius_km
            within_cap = dist_km <= MAX_CIRCLE_RATE_FALLBACK_KM
            filled_this = int(within_cap.sum())
            if filled_this > 0:
                t_arr = target_index.to_numpy()
                df.loc[t_arr[within_cap], "circle_rate"] = (
                    df.loc[nearest_index[within_cap], "circle_rate"].to_numpy()
                )
                df.loc[t_arr[within_cap], "_circle_rate_fill_dist_km"] = dist_km[within_cap]
            filled_rows += filled_this

        remaining_missing = int((df["circle_rate"].isna() & valid_coord_mask).sum())
        if filled_rows > 0 and "_circle_rate_fill_dist_km" in df.columns:
            median_fill_distance_km = float(df["_circle_rate_fill_dist_km"].median())
            logging.info(
                "Haversine circle-rate fallback filled rows=%s, "
                "remaining_missing_with_coords=%s, median_distance_km=%.2f",
                filled_rows,
                remaining_missing,
                median_fill_distance_km,
            )
        else:
            logging.info("Haversine circle-rate fallback filled rows=0")

        return df.drop(
            columns=["_circle_prop_type", "_circle_city_key", "_circle_rate_fill_dist_km"],
            errors="ignore",
        )

    def _save_missing_circle_rate_json(self, df: pd.DataFrame) -> None:
        """
        Save unresolved (city, locality) pairs to JSON on every run.

        File format is city -> {locality: rate_or_null}. Any numeric values already
        present in the JSON are preserved as manual overrides for future runs.
        """
        try:
            path = os.path.join(CIRCLE_RATES_DIR, "missing_circle_rates.json")
            os.makedirs(CIRCLE_RATES_DIR, exist_ok=True)

            # Keep previously filled numeric overrides so user updates are not lost.
            preserved_overrides: dict[str, dict[str, float]] = {}
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    existing = json.load(fh)
                if isinstance(existing, dict):
                    for city, loc_map in existing.items():
                        if str(city).startswith("_") or not isinstance(loc_map, dict):
                            continue
                        if _normalize_city_name(city) in _EXCLUDED_CITY_NAMES:
                            continue
                        for locality, rate in loc_map.items():
                            if locality is None:
                                continue
                            parsed_rate = None
                            if isinstance(rate, (int, float)) and not pd.isna(rate):
                                parsed_rate = float(rate)
                            elif isinstance(rate, str):
                                try:
                                    parsed_rate = float(rate.strip())
                                except Exception:
                                    parsed_rate = None
                            if parsed_rate is not None:
                                city_s = str(city).strip()
                                loc_s = str(locality).strip()
                                if city_s and loc_s:
                                    preserved_overrides.setdefault(city_s, {})[loc_s] = parsed_rate

            # Build current unresolved list from this run.
            missing_rows = df[df["circle_rate"].isna()].copy()
            missing_pairs = missing_rows[
                missing_rows["city"].notna() & missing_rows["locality"].notna()
            ][["city", "locality"]].copy()

            if not missing_pairs.empty:
                missing_pairs["city"] = missing_pairs["city"].astype(str).str.strip()
                missing_pairs["locality"] = missing_pairs["locality"].astype(str).str.strip()
                missing_pairs["city_norm"] = missing_pairs["city"].map(_normalize_city_name)
                missing_pairs = missing_pairs[
                    (missing_pairs["city"] != "")
                    & (missing_pairs["locality"] != "")
                    & (missing_pairs["city"].str.lower() != "nan")
                    & (missing_pairs["locality"].str.lower() != "nan")
                    & (~missing_pairs["city_norm"].isin(_EXCLUDED_CITY_NAMES))
                ]
                missing_pairs = missing_pairs.drop(columns=["city_norm"], errors="ignore")

            missing_counts = {}
            if not missing_pairs.empty:
                grouped = (
                    missing_pairs.groupby(["city", "locality"], as_index=False)
                    .size()
                    .rename(columns={"size": "missing_count"})
                )
                for _, row in grouped.sort_values(["city", "locality"]).iterrows():
                    city_s = row["city"]
                    loc_s = row["locality"]
                    missing_counts.setdefault(city_s, {})[loc_s] = int(row["missing_count"])

            # Final payload: preserve numeric overrides + add current missing as null.
            payload_city_map = {
                city: dict(sorted(loc_map.items(), key=lambda x: x[0].lower()))
                for city, loc_map in preserved_overrides.items()
            }
            for city, loc_map in missing_counts.items():
                city_bucket = payload_city_map.setdefault(city, {})
                for locality in loc_map.keys():
                    city_bucket.setdefault(locality, None)

            payload_city_map = dict(
                sorted(payload_city_map.items(), key=lambda x: x[0].lower())
            )

            total_rows = len(df)
            missing_rows_count = int(df["circle_rate"].isna().sum())
            matched_rows_count = total_rows - missing_rows_count
            unique_missing = sum(len(v) for v in missing_counts.values())
            override_count = sum(len(v) for v in preserved_overrides.values())

            payload = {
                "_generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "_total_rows": int(total_rows),
                "_matched_rows": int(matched_rows_count),
                "_missing_rows": int(missing_rows_count),
                "_unique_missing_city_locality": int(unique_missing),
                "_preserved_numeric_overrides": int(override_count),
                "_note": "Fill numeric INR/sqft values for null localities. Numeric values are reused in future runs.",
            }
            payload.update(payload_city_map)

            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)

            logging.info(
                "  missing_circle_rates.json saved -> %s "
                "(missing_rows=%s, unique_missing=%s, preserved_overrides=%s)",
                path,
                missing_rows_count,
                unique_missing,
                override_count,
            )
        except Exception as e:
            logging.warning(f"  Could not save missing_circle_rates.json: {e}")

    def _add_circle_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 10: Add circle_rate column")

        if self.circle_rate_matcher is None:
            logging.warning("  Circle rate matcher not available, filling NaN")
            df["circle_rate"] = np.nan
        else:
            df["circle_rate"] = df.apply(
                lambda row: self.circle_rate_matcher.get_rate(
                    row.get("city", ""), row.get("locality", "")
                ),
                axis=1,
            )

            matched = df["circle_rate"].notna().sum()
            total = len(df)
            logging.info(f"  Circle rate matched: {matched}/{total} ({matched/total:.1%})")

            # Per-city breakdown
            if "city" in df.columns:
                for city, grp in df.groupby("city"):
                    m = grp["circle_rate"].notna().sum()
                    t = len(grp)
                    logging.info(f"    {city}: {m}/{t} ({m/t:.1%})")

        # Save unmatched localities BEFORE haversine so the JSON reflects genuine
        # text-match failures (not haversine-imputed ones).
        self._save_missing_circle_rate_json(df)

        # Fill remaining missing values via nearest lat/lon (haversine BallTree)
        before_haversine = int(df["circle_rate"].isna().sum())
        if before_haversine > 0:
            logging.info(f"  Attempting haversine fill for {before_haversine} unmatched rows "
                         f"(max distance {MAX_CIRCLE_RATE_FALLBACK_KM} km)...")
            df = self._fill_circle_rate_by_nearest_haversine(df)
            after_haversine = int(df["circle_rate"].isna().sum())
            logging.info(
                f"  After haversine fill: {after_haversine} rows still missing "
                f"(filled {before_haversine - after_haversine}, "
                f"beyond {MAX_CIRCLE_RATE_FALLBACK_KM} km cap or no donor)"
            )
            if after_haversine > 0:
                before_drop = len(df)
                df = df[df["circle_rate"].notna()].reset_index(drop=True)
                logging.info(
                    f"  Dropped {before_drop - len(df)} rows with no circle_rate "
                    f"within {MAX_CIRCLE_RATE_FALLBACK_KM} km. Remaining: {len(df)}"
                )

        return df

    # ============================================================
    #  STEP 11 – final locality null cleanup
    # ============================================================
    def _drop_missing_locality_final(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final cleanup after all feature engineering:
        drop rows where locality is null/blank.
        """
        logging.info("Step 11: Final locality cleanup")

        if "locality" not in df.columns:
            logging.warning("  locality column not found, skipping final locality cleanup")
            return df

        before = len(df)
        locality_norm = df["locality"].fillna("").astype(str).str.strip()
        invalid_locality = locality_norm.eq("") | locality_norm.str.lower().eq("nan")

        dropped = int(invalid_locality.sum())
        if dropped > 0:
            df = df.loc[~invalid_locality].copy()

        logging.info(f"  Dropped rows with missing locality: {dropped} ({before} -> {len(df)})")
        return df

    # ============================================================
    #  STEP 12 – final property-type specific column cleanup
    # ============================================================
    def _final_property_type_column_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final cleanup requested for land/plot vs non-plot rows:
          - Null out is_gated for plot/land rows
          - Null out road_width/road_width_sqft/road_width_ft/gated_plot for non-plot rows
        """
        logging.info("Step 12: Final property-type specific column cleanup")

        if "property_type_grouped" not in df.columns:
            logging.warning("  property_type_grouped not found, skipping property-type cleanup")
            return df

        prop_key = df["property_type_grouped"].fillna("").astype(str).str.strip().str.lower()
        plot_land_mask = prop_key.isin({"plot", "land"})
        non_plot_mask = ~plot_land_mask

        if "is_gated" in df.columns:
            affected = int(df.loc[plot_land_mask, "is_gated"].notna().sum())
            if affected > 0:
                df.loc[plot_land_mask, "is_gated"] = np.nan
            logging.info(f"  is_gated nulled for plot/land rows: {affected}")

        for col in ["road_width", "road_width_sqft", "road_width_ft", "gated_plot"]:
            if col not in df.columns:
                continue
            affected = int(df.loc[non_plot_mask, col].notna().sum())
            if affected > 0:
                df.loc[non_plot_mask, col] = np.nan
            logging.info(f"  {col} nulled for non-plot rows: {affected}")

        return df

    # ============================================================
    #  MAIN ENTRYPOINT
    # ============================================================
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("=" * 60)
            logging.info("DATA TRANSFORMATION STARTED")
            logging.info("=" * 60)

            df = pd.read_csv(self.data_ingestion_artifact.merged_file_path, low_memory=False)
            logging.info(f"Loaded merged data: shape={df.shape}")

            df = self._coerce_numeric_columns(
                df,
                [
                    "covered_area_value",
                    "price_numeric",
                    "latitude",
                    "longitude",
                    "bhk",
                    "bathrooms",
                    "balconies",
                    "corner_property",
                    "rectangular_plot",
                    "gated_plot",
                    "backlane",
                ],
                "initial load",
            )

            df = self._clean(df)
            df = self._remove_excluded_cities(df)
            df = self._drop_columns(df)
            df = self._fill_from_description(df)
            df = self._transform_age(df)
            df = self._convert_area_to_sqft(df)
            df = self._furnishing_dummies(df)
            df = self._property_type_dummies(df)
            df = self._add_price_per_sqft(df)
            df = self._add_boolean_features(df)
            df = self._standardise_facing(df)
            df = self._standardise_possession(df)
            df = self._fill_locality(df)
            df = self._add_circle_rate(df)
            df = self._drop_missing_locality_final(df)
            df = self._final_property_type_column_cleanup(df)

            # Save
            os.makedirs(self.config.transformed_data_dir, exist_ok=True)
            
            # Try to save, handle locked file
            output_path = self.config.transformed_file_path
            try:
                df.to_csv(output_path, index=False)
                logging.info(f"Transformed data saved -> {output_path}")
            except PermissionError:
                # File is locked, save with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.config.transformed_file_path.replace('.csv', f'_{timestamp}.csv')
                df.to_csv(output_path, index=False)
                logging.warning(f"Original file locked, saved to -> {output_path}")
            
            logging.info(f"Final shape: {df.shape}")
            logging.info(f"Columns: {df.columns.tolist()}")

            # ── Split by property type into cleaned_data/ ────────
            cleaned_data_dir = os.path.join(self.config.transformed_data_dir, "cleaned_data")
            os.makedirs(cleaned_data_dir, exist_ok=True)

            ohe_cols = [c for c in df.columns if c.startswith("prop_type_")]
            drop_for_split = ["property_type_grouped"] + ohe_cols

            # User-requested export cleanup for cleaned_data folder only.
            # This does not affect cleaned.csv or merged.csv.
            _CLEANED_FOLDER_DROP_COLS = [
                "id",
                "event_type",
                "covered_area_value",
                "covered_area_unit",
                "price_raw",
                "sqft_price",
                "property_type",
                "description",
                "address_full",
                "amenities",
                "agent_id",
                "agent_name",
                "agent_type",
                "developer_name",
                "developer_id",
                "posting_date",
                "scrape_date",
                "company_name",
                "project_society_name",
                "desc_bhk",
                "desc_balconies",
                "desc_price",
                "desc",
                "desc_area",
                "desc_unit",
                "desc_area_sqft",
            ]

            # Columns irrelevant for plots (no rooms / furnishing / floors)
            _PLOT_EXTRA_DROP = [
                "bhk", "balconies", "bathrooms", "floors",
                "furnishing_type", "possession_status",
                "desc_bhk", "desc_bathrooms", "desc_balconies",
                "is_gated",
            ]

            # Columns to remove for all non-plot/non-land property files.
            _NON_PLOT_EXTRA_DROP = [
                "road_width", "road_width_sqft", "road_width_ft", "gated_plot",
            ]

            for prop_type, group_df in df.groupby("property_type_grouped"):
                prop_type_key = str(prop_type).strip().lower()
                is_plot_land = prop_type_key in {"plot", "land"}
                extra = _PLOT_EXTRA_DROP if is_plot_land else _NON_PLOT_EXTRA_DROP

                split_df = group_df.copy()

                # Apply price filter only for cleaned_data exports.
                if "price_per_sqft" in split_df.columns:
                    split_df["price_per_sqft"] = pd.to_numeric(split_df["price_per_sqft"], errors="coerce")
                    before_filter = len(split_df)
                    ppsf_min = self.config.price_per_sqft_min
                    split_df = split_df[split_df["price_per_sqft"] >= ppsf_min].copy()
                    removed = before_filter - len(split_df)
                    logging.info(f"  {prop_type}: removed {removed} rows with price_per_sqft < {ppsf_min}")
                else:
                    logging.warning(f"  {prop_type}: price_per_sqft not found, skipping >=800 filter")

                dynamic_drop_cols = [
                    col
                    for col in split_df.columns
                    if col.startswith("desc_") or col.startswith("project_society")
                ]

                split_df = split_df.drop(
                    columns=drop_for_split + extra + _CLEANED_FOLDER_DROP_COLS + dynamic_drop_cols,
                    errors="ignore",
                )
                split_path = os.path.join(cleaned_data_dir, f"{prop_type}.csv")
                try:
                    split_df.to_csv(split_path, index=False)
                    logging.info(f"  Saved {prop_type}: {len(split_df)} rows -> {split_path}")
                except PermissionError:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    split_path_locked = os.path.join(cleaned_data_dir, f"{prop_type}_{timestamp}.csv")
                    split_df.to_csv(split_path_locked, index=False)
                    logging.warning(
                        f"  {prop_type} file was locked, saved {len(split_df)} rows -> {split_path_locked}"
                    )

            logging.info("=" * 60)
            logging.info("DATA TRANSFORMATION COMPLETED")
            logging.info("=" * 60)

            return DataTransformationArtifact(
                transformed_file_path=output_path,
            )

        except Exception as e:
            raise RealEstateException(e, sys)
