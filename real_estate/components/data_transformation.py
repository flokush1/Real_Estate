import os
import re
import sys
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
from real_estate.constant import TARGET_COLUMN, CITY_LOCALITIES_JSON
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
from real_estate.utils.locality_matcher import LocalityMatcher


# ────────────────────────────────────────────────────────────────
#  AREA CONVERSION FACTORS  →  everything to Sq-ft
# ────────────────────────────────────────────────────────────────
_SQFT_CONVERSION = {
    "sq-ft": 1.0,
    "sq.ft": 1.0,
    "sqft": 1.0,
    "sq-yrd": 9.0,
    "sq.yd": 9.0,
    "sq-m": 10.7639,
    "acre": 43560.0,
    "bigha": 27000.0,
    "hectare": 107639.0,
    "marla": 272.25,
    "ground": 2400.0,
    "rood": 10890.0,
    "biswa": 1350.0,
    "biswa1": 1350.0,
    "biswa2": 1350.0,
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
    #  STEP 2 – extract values from description & replace originals
    # ============================================================
    def _fill_from_description(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Step 2: Extract values from description/amenities & replace originals")
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
        
        # Convert balconies to numeric first
        df["balconies"] = pd.to_numeric(df["balconies"], errors="coerce")
        
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
        
        original_price_na = df["price_numeric"].isna().sum()
        df["price_numeric"] = df["desc_price"].fillna(df["price_numeric"])
        logging.info(f"    price_numeric: replaced with desc_price (filled {df['price_numeric'].notna().sum() - (len(df) - original_price_na)} additional)")
        
        # For area: use desc_area_sqft (converted), fallback to original covered_area_value
        # Also update unit to "Sq-ft" only where we used desc_area_sqft
        original_area_na = df["covered_area_value"].isna().sum()
        used_desc_mask = df["desc_area_sqft"].notna()
        df.loc[used_desc_mask, "covered_area_value"] = df.loc[used_desc_mask, "desc_area_sqft"]
        df.loc[used_desc_mask, "covered_area_unit"] = "Sq-ft"
        # For rows without desc_area, keep original covered_area_value and unit
        logging.info(f"    covered_area_value: replaced {used_desc_mask.sum()} values with desc_area_sqft (in Sq-ft)")
        logging.info(f"    covered_area_unit: set to Sq-ft for {used_desc_mask.sum()} rows from desc")
        
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
            factor = _SQFT_CONVERSION.get(str(unit).lower().strip(), None)
            if factor is None:
                logging.warning(f"  Unknown unit '{unit}', keeping as-is")
                return val
            return val * factor

        df["covered_area_sqft"] = df.apply(_to_sqft, axis=1)
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

        # Apply city-aware locality extraction
        def extract_locality_row(row):
            return self.locality_matcher.extract_locality(
                city=row.get('city', ''),
                description=row.get('description', ''),
                address=row.get('address_full', ''),
                current_locality=row.get('locality', ''),
            )

        df['locality'] = df.apply(extract_locality_row, axis=1)

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
    #  MAIN ENTRYPOINT
    # ============================================================
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("=" * 60)
            logging.info("DATA TRANSFORMATION STARTED")
            logging.info("=" * 60)

            df = pd.read_csv(self.data_ingestion_artifact.merged_file_path)
            logging.info(f"Loaded merged data: shape={df.shape}")

            df = self._clean(df)
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

            # Columns irrelevant for plots (no rooms / furnishing / floors)
            _PLOT_EXTRA_DROP = [
                "bhk", "balconies", "bathrooms", "floors",
                "furnishing_type", "possession_status",
                "desc_bhk", "desc_bathrooms", "desc_balconies",
            ]

            for prop_type, group_df in df.groupby("property_type_grouped"):
                extra = _PLOT_EXTRA_DROP if prop_type == "plot" else []
                split_df = group_df.drop(columns=drop_for_split + extra, errors="ignore")
                split_path = os.path.join(cleaned_data_dir, f"{prop_type}.csv")
                split_df.to_csv(split_path, index=False)
                logging.info(f"  Saved {prop_type}: {len(split_df)} rows -> {split_path}")

            logging.info("=" * 60)
            logging.info("DATA TRANSFORMATION COMPLETED")
            logging.info("=" * 60)

            return DataTransformationArtifact(
                transformed_file_path=output_path,
            )

        except Exception as e:
            raise RealEstateException(e, sys)
