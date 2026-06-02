import importlib
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import BallTree, KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from real_estate.constant import CIRCLE_RATES_DIR, NCR_ROADS_GEOJSON_PATH, ROADS_GEOJSON_PATHS
from real_estate.entity import (
    PlotDataIngestionArtifact,
    PlotDataTransformationArtifact,
    PlotDataTransformationConfig,
)
from real_estate.exception.exception import RealEstateException
from real_estate.logging.logger import logging
from real_estate.utils.circle_rate_matcher import CircleRateMatcher, _resolve_city


SQFT_CONVERSION = {
    "sq-ft": 1.0,
    "sqft": 1.0,
    "sq.ft": 1.0,
    "sq-yrd": 9.0,
    "sq.yd": 9.0,
    "sq-m": 10.7639,
    "sq.mt": 10.7639,
    "acre": 43560.0,
    "guntha": 1089.0,   # 1 Guntha = 1/40 acre = 1089 sq ft (Maharashtra/Goa)
    "gunta": 1089.0,    # alternate spelling
}

AMENITY_RULES = [
    (
        "park",
        [
            r"(?:near|facing|overlook|adjacent|close to|proximity to|access to)\s+\w*\s*park\b",
            r"\bpark\s*facing\b",
            r"\bgarden/park\b",
            r"garden.*?park",
            r"\bopen\s+park\b",
            r"\bright.*?park\b",
            r"\bexpansive park\b",
            r"\blarge park\b",
        ],
    ),
    ("school", [r"\bschool\b", r"\bvidyalaya\b", r"\beducational institution\b"]),
    (
        "hospital",
        [
            r"\bhospital\b",
            r"\bclinic\b",
            r"\bmedical cent(?:re|er)\b",
            r"\bnursing home\b",
            r"\bdiagnostic cent(?:re|er)\b",
        ],
    ),
    (
        "market",
        [
            r"\bmarket\b",
            r"\bshopping mall\b",
            r"\bshopping cent(?:re|er)\b",
            r"\blocal market\b",
            r"\bhuda market\b",
        ],
    ),
    (
        "metro",
        [
            r"\bmetro\s+station\b",
            r"\bmetro\s+connectivity\b",
            r"\bnear\s+metro\b",
            r"\bmetro\s+line\b",
            r"\bupcoming\s+metro\b",
        ],
    ),
    ("airport", [r"\bairport\b"]),
    (
        "highway",
        [r"\bhighway\b", r"\bexpressway\b", r"\bouter ring road\b", r"\bperipheral\b", r"\bnh[-\s]?\d+\b"],
    ),
    ("water_supply", [r"\bwater supply\b", r"\bwater connection\b", r"\b24.hour water\b", r"\bwater\s+availability\b"]),
    ("electricity", [r"\belectricity\s+connection\b", r"\bpower supply\b", r"\bpower connection\b", r"\belectricity\s+available\b"]),
    ("security", [r"\bsecurity\b", r"\bcctv\b", r"\bguard\b", r"\bsurveillance\b"]),
    ("gymnasium", [r"\bgymnasium\b", r"\bgym\s+facilit", r"\bfitness cent(?:re|er)\b"]),
    ("swimming_pool", [r"\bswimming pool\b"]),
    ("club_house", [r"\bclub\s*house\b", r"\bcommunity hall\b", r"\bcommunity cent(?:re|er)\b"]),
    ("boundary_wall", [r"\bboundary wall\b"]),
    ("sewerage", [r"\bsewerage\b", r"\bdrainage\b", r"\bsewer\b"]),
    ("college", [r"\bcollege\b", r"\buniversity\b"]),
    ("bank_atm", [r"\batm\b", r"\bbank branch\b", r"\bnear.*?bank\b"]),
    ("restaurant", [r"\brestaurant\b", r"\bcafeteria\b"]),
    ("temple", [r"\btemple\b", r"\bmandir\b", r"\bmasjid\b", r"\bchurch\b"]),
    ("vastu", [r"\bvastu\b"]),
    ("street_light", [r"\bstreet light\b", r"\bstreet lighting\b"]),
    ("wide_road", [r"\bwide\s+road\b", r"\bwide\s+(?:clean\s+)?roads\b", r"\bpaved road\b"]),
]

AMENITY_TOKEN_MAP = {
    "boundary wall": "boundary_wall",
    "garden/park": ["park", "garden"],
    "park": "park",
    "garden": "garden",
    "children's play area": "playground",
    "children play area": "playground",
    "gated community": "gated_community",
    "24x7 water supply": "water_supply",
    "24x7water supply": "water_supply",
    "ro water system": "water_supply",
    "gymnasium": "gymnasium",
    "gym": "gymnasium",
    "24x7 security": "security",
    "security guards": "security",
    "security guard": "security",
    "security": "security",
    "24x7 cctv surveillance": "security",
    "cctv": "security",
    "power backup": "power_backup",
    "club house": "club_house",
    "clubhouse": "club_house",
    "swimming pool": "swimming_pool",
    "car parking": "parking",
    "parking": "parking",
    "atm": "atm_bank",
    "restaurant": "restaurant",
    "jogging track": "jogging_track",
    "jogging and strolling track": "jogging_track",
    "cycling & jogging track": "jogging_track",
    "cycling track": "jogging_track",
    "cycling and jogging track": "jogging_track",
    "vastu compliant": "vastu_compliant",
    "main road": "wide_road",
    "internal roads & footpaths": "wide_road",
    "internal road": "wide_road",
    "outdoor tennis courts": "sports_court",
    "tennis court": "sports_court",
    "badminton court": "sports_court",
    "community hall": "community_hall",
    "banquet hall": "community_hall",
    "landscaping & tree planting": "landscaping",
    "landscaping": "landscaping",
    "street lighting": "street_lighting",
    "rain water harvesting": "rainwater_harvesting",
    "sewage treatment plant": "sewage_treatment",
    "storm water drains": "storm_drains",
    "intercom facility": "intercom",
    "fire fighting system": "fire_fighting",
    "yoga / meditation area": "yoga_meditation",
    "yoga/meditation area": "yoga_meditation",
    "maintenance staff": "maintenance",
}


class PlotDataTransformation:
    """
    Plot/Land data preprocessing component adapted from clean_plot.py.

    Pipeline summary:
    - Extract structured fields from description
    - Standardize areas, prices, facing, amenities
    - Merge source + description derived fields
    - Filter outliers
    - KNN imputation for facing and road_width_m
    - Add circle rates and road-distance features
    - Build final model-ready schema
    """

    def __init__(
        self,
        data_ingestion_artifact: PlotDataIngestionArtifact,
        config: PlotDataTransformationConfig = PlotDataTransformationConfig(),
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.config = config

        try:
            self.circle_rate_matcher = CircleRateMatcher(CIRCLE_RATES_DIR)
            logging.info(
                f"PlotDataTransformation loaded circle-rate entries={self.circle_rate_matcher.total_entries}"
            )
        except Exception as e:
            logging.warning(f"Could not initialize CircleRateMatcher: {e}")
            self.circle_rate_matcher = None

    @staticmethod
    def _merge_numeric(primary, secondary):
        if pd.isna(primary) and pd.isna(secondary):
            return np.nan
        if pd.isna(primary):
            return secondary
        return primary

    @staticmethod
    def _merge_categorical(primary, secondary):
        p_null = pd.isna(primary) or str(primary).strip() == ""
        s_null = pd.isna(secondary) or str(secondary).strip() == ""
        if p_null and s_null:
            return np.nan
        if p_null:
            return secondary
        return primary

    @staticmethod
    def _merge_bool(primary, secondary):
        p = 0 if pd.isna(primary) else int(primary)
        s = 0 if pd.isna(secondary) else int(secondary)
        return 1 if (p == 1 or s == 1) else 0

    @staticmethod
    def _convert_to_sqft(value, unit):
        if pd.isna(value) or pd.isna(unit):
            return value
        factor = SQFT_CONVERSION.get(str(unit).strip().lower())
        if factor is None:
            return value
        return round(value * factor, 2)

    @staticmethod
    def _standardize_facing(val):
        if pd.isna(val):
            return val
        s = str(val).strip()
        s = re.sub(r"^the\s+", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*facing$", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*[-–]\s*", " ", s)
        s = re.sub(r"\s+", " ", s).strip().title()
        return s

    @staticmethod
    def _extract_des_features(desc):
        if not isinstance(desc, str):
            return {
                "des_covered_area_value": np.nan,
                "des_covered_area_unit": np.nan,
                "des_price_numeric": np.nan,
                "des_property_type": np.nan,
                "des_facing_direction": np.nan,
                "des_corner_property": np.nan,
                "des_rectangular_plot": np.nan,
                "des_road_width": np.nan,
                "des_gated_plot": np.nan,
                "des_backlane": np.nan,
                "des_allowed_floors": np.nan,
                "des_park_facing": np.nan,
                "des_tenure_type": np.nan,
                "des_property_status": np.nan,
                "des_rera_approved": np.nan,
            }

        d = desc.strip()

        area_val, area_unit = np.nan, np.nan
        m = re.search(
            r"(\d+\.?\d*)\s+(?:Square\s+(meters?|yards?|feet?)|square_(meter|yard|feet))\s+Plot",
            d,
            re.IGNORECASE,
        )
        if m:
            area_val = float(m.group(1))
            raw_unit = (m.group(2) or m.group(3) or "").lower().rstrip("s")
            area_unit = {
                "meter": "Sq-m",
                "yard": "Sq-yrd",
                "feet": "Sq-ft",
                "foot": "Sq-ft",
            }.get(raw_unit, raw_unit)

        if np.isnan(area_val):
            m = re.search(
                r"(\d+\.?\d*)\s+(Sq-ft|Sq-yrd|Sq-m|Acre)\s+"
                r"(?:Residential Plot|Commercial Land|Agricultural Land|Industrial Land)",
                d,
                re.IGNORECASE,
            )
            if m:
                area_val = float(m.group(1))
                area_unit = m.group(2)

        if np.isnan(area_val):
            m = re.search(
                r"(\d+\.?\d*)\s+(Acre)\s+(?:Residential|Commercial|Agricultural|Industrial)",
                d,
                re.IGNORECASE,
            )
            if m:
                area_val = float(m.group(1))
                area_unit = "Acre"

        price = np.nan
        m = re.search(r"\bSale Price\s+(\d+)", d)
        if m:
            price = float(m.group(1))

        if np.isnan(price):
            m = re.search(
                r"available at a price of Rs\s+(\d+\.?\d*)\s*(Cr|Crore|L|Lac|Lakh)?",
                d,
                re.IGNORECASE,
            )
            if m:
                val = float(m.group(1))
                unit_p = (m.group(2) or "").lower()
                if unit_p in ("cr", "crore"):
                    price = val * 1e7
                elif unit_p in ("l", "lac", "lakh"):
                    price = val * 1e5
                else:
                    price = val

        if np.isnan(price):
            m = re.search(r"for\s+(\d+\.?\d*)\s*(Crore|Cr|Lac|Lakh|L)\(?s?\)?", d, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                unit_p = m.group(2).lower()
                price = val * 1e7 if unit_p in ("cr", "crore") else val * 1e5

        prop_type = np.nan
        m = re.search(
            r"\b(Residential Plot|Commercial Land|Agricultural Land|Industrial Land)\b",
            d,
            re.IGNORECASE,
        )
        if m:
            prop_type = m.group(1).title()

        facing = np.nan
        m = re.search(
            r"The plot is\s+(North[\s\-]+East|North[\s\-]+West|South[\s\-]+East|South[\s\-]+West|"
            r"North|South|East|West)\s+facing",
            d,
            re.IGNORECASE,
        )
        if m:
            facing = re.sub(r"[\s\-]+", "-", m.group(1).strip().title())

        if not isinstance(facing, str):
            m = re.search(
                r"\b(North - East|North - West|South - East|South - West|North|South|East|West|Central)\s+Sale Price",
                d,
                re.IGNORECASE,
            )
            if m:
                facing = re.sub(r"\s*-\s*", "-", m.group(1).strip().title())

        if not isinstance(facing, str):
            m = re.search(
                r"\b(north[\s\-]+east|north[\s\-]+west|south[\s\-]+east|south[\s\-]+west|"
                r"north|south|east|west)[\s\-]+facing\b",
                d,
                re.IGNORECASE,
            )
            if m:
                facing = re.sub(r"[\s\-]+", "-", m.group(1).strip().title())

        if not isinstance(facing, str):
            m = re.search(
                r"\b(North - East|North - West|South - East|South - West|North|South|East|West)\s+"
                r"(?:Facing\s+)?(?:Freehold|Leasehold|Resale|New Property)",
                d,
                re.IGNORECASE,
            )
            if m:
                facing = re.sub(r"\s*-\s*", "-", m.group(1).strip().title())

        if not isinstance(facing, str):
            facing = np.nan

        corner = 1 if re.search(r"\bcorner\b", d, re.IGNORECASE) else 0
        rectangular = 1 if re.search(r"\brectangular\b", d, re.IGNORECASE) else 0

        road_w = np.nan
        m = re.search(r"width of the facing road is\s+(\d+\.?\d*)\s*mt", d, re.IGNORECASE)
        if m:
            road_w = float(m.group(1))

        if np.isnan(road_w):
            m = re.search(
                r"road of\s+(\d+\.?\d*)\s*(?:m|mtr|meter|metre)s?"
                r"|(?:facing\s+)?(\d+\.?\d*)\s*(?:m|mtr|meter|metre)s?\s*(?:wide\s+)?road"
                r"|(\d+\.?\d*)[-\s]?(?:meter|metre|mtr)s?[-\s]?road",
                d,
                re.IGNORECASE,
            )
            if m:
                road_w = float(next(v for v in m.groups() if v is not None))

        gated = 1 if re.search(
            r"gated\s+(?:community|locality|enclave|society|plot|area|complex|development)|"
            r"Property in a Gated Locality|\bgated\b",
            d,
            re.IGNORECASE,
        ) else 0
        backlane = 1 if re.search(r"back[\s\-]?lane", d, re.IGNORECASE) else 0

        floors = np.nan
        m = re.search(r"maximum floor allowed for construction on this plot is\s+(\d+)", d, re.IGNORECASE)
        if m:
            floors = int(m.group(1))

        if np.isnan(floors):
            floor_pats = [
                r"allows the buyer to build\s+(\d+)\s+of floors",
                r"land allows the buyer to build\s+(\d+)",
                r"permitted to construct\s+(\d+)\s+for construction floors",
                r"permits up to\s+(\d+)\s+floors",
                r"allows construction of up to\s+(\d+)\s+floors",
                r"construction of up to\s+(\d+)\s+floors",
                r"potential with\s+(\d+)\s+floors allowed",
                r"(\d+)\s+floors allowed for the plot",
                r"(\d+)\s+for construction floors",
                r"construct\s+(\d+)\s+(?:of\s+)?floors",
                r"up to\s+(\d+)\s+floors",
            ]
            for pat in floor_pats:
                m = re.search(pat, d, re.IGNORECASE)
                if m:
                    floors = int(m.group(1))
                    break

        if np.isnan(floors):
            m = re.search(r"up to\s+(one|two|three|four|five|six)\s+floors", d, re.IGNORECASE)
            if m:
                floors = {
                    "one": 1,
                    "two": 2,
                    "three": 3,
                    "four": 4,
                    "five": 5,
                    "six": 6,
                }[m.group(1).lower()]

        park_facing = 1 if re.search(
            r"park\s*facing|facing\s+(?:a\s+)?park|park\s+view|garden/park|overlooks.*?(?:park|garden)",
            d,
            re.IGNORECASE,
        ) else 0

        tenure = np.nan
        m = re.search(r"\b(Freehold|Leasehold)\b", d, re.IGNORECASE)
        if m:
            tenure = m.group(1).title()

        status = np.nan
        m = re.search(r"\b(New Property|Resale)\b", d, re.IGNORECASE)
        if m:
            status = "New" if "new" in m.group(1).lower() else "Resale"

        rera = 1 if re.search(r"\bRERA\b", d, re.IGNORECASE) else 0

        return {
            "des_covered_area_value": area_val,
            "des_covered_area_unit": area_unit,
            "des_price_numeric": price,
            "des_property_type": prop_type,
            "des_facing_direction": facing,
            "des_corner_property": corner,
            "des_rectangular_plot": rectangular,
            "des_road_width": road_w,
            "des_gated_plot": gated,
            "des_backlane": backlane,
            "des_allowed_floors": floors,
            "des_park_facing": park_facing,
            "des_tenure_type": tenure,
            "des_property_status": status,
            "des_rera_approved": rera,
        }

    @staticmethod
    def _extract_des_amenities(desc):
        if not isinstance(desc, str) or not desc.strip():
            return np.nan

        d = desc.lower()
        found = []

        m = re.search(
            r"most popular landmarks near this plot are\s+(.+?)(?:\s{2,}|Residential Plot|Property in|$)",
            desc,
            re.IGNORECASE,
        )
        if m:
            raw_landmarks = m.group(1).strip().rstrip(".")
            landmarks = [lm.strip() for lm in raw_landmarks.split(",") if 2 < len(lm.strip()) < 60]
            found.extend(landmarks[:5])

        for canonical, patterns in AMENITY_RULES:
            for pat in patterns:
                if re.search(pat, d):
                    if canonical not in found:
                        found.append(canonical)
                    break

        return ", ".join(found) if found else np.nan

    @staticmethod
    def _parse_amenities_col(val):
        if pd.isna(val):
            return set()
        keywords = set()
        for tok in str(val).split(","):
            key = tok.split(":")[0].strip().lower()
            mapped = AMENITY_TOKEN_MAP.get(key)
            if mapped:
                if isinstance(mapped, list):
                    keywords.update(mapped)
                else:
                    keywords.add(mapped)
        return keywords

    @staticmethod
    def _parse_des_amenities(val):
        if pd.isna(val) or str(val).strip() == "":
            return set()
        return {t.strip() for t in str(val).split(",") if t.strip()}

    @classmethod
    def _merge_amenities(cls, structured_val, des_val):
        kw = cls._parse_amenities_col(structured_val) | cls._parse_des_amenities(des_val)
        return ", ".join(sorted(kw)) if kw else np.nan

    @staticmethod
    def _has_kw(amenity_val, keyword):
        if pd.isna(amenity_val):
            return False
        return keyword in [t.strip() for t in str(amenity_val).split(",")]

    @staticmethod
    def _parse_road_width_m(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower().replace("\n", " ")
        if s in {"", "no info", "nan", "none"}:
            return np.nan
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        if not m:
            return np.nan
        num = float(m.group(1))
        if re.search(r"\bft\b|feet|foot", s):
            num = round(num * 0.3048, 2)
        if num < 1 or num > 100:
            return np.nan
        return num

    @staticmethod
    def _classify_road_class(ref_value, fclass_value=None):
        if not pd.isna(ref_value) and str(ref_value).strip():
            ref = str(ref_value).upper()
            compact = re.sub(r"\s+", "", ref)
            if compact.startswith("NH") or re.search(r"\bNH\b", ref):
                return "NH"
            if compact.startswith("SH") or re.search(r"\bSH\b", ref):
                return "SH"
            if compact.startswith("MDR") or re.search(r"\bMDR\b", ref):
                return "MDR"
        # fclass-based fallback for roads without a ref (Pune/Jaipur etc.)
        if fclass_value and not pd.isna(fclass_value):
            fc = str(fclass_value).lower().strip()
            if fc in ("motorway", "trunk"):
                return "NH"
            if fc in ("primary", "primary_link"):
                return "SH"
            if fc in ("secondary", "secondary_link"):
                return "MDR"
        return None

    @staticmethod
    def _nearest_distance_km(points_proj, roads_proj_subset, gpd):
        if roads_proj_subset.empty:
            return pd.Series(np.nan, index=points_proj.index, dtype=float)

        try:
            joined = gpd.sjoin_nearest(
                points_proj,
                roads_proj_subset[["geometry"]],
                how="left",
                distance_col="_dist_m",
            )
            dist_m = (
                joined["_dist_m"].astype(float).groupby(level=0).min().reindex(points_proj.index)
            )
        except Exception:
            merged = roads_proj_subset.geometry.unary_union
            dist_m = points_proj.geometry.distance(merged).astype(float)

        return dist_m / 1000.0

    @staticmethod
    def _add_road_width_flags(data: pd.DataFrame, road_width_col: str = "road_width_m") -> pd.DataFrame:
        data = data.copy()
        road_width = pd.to_numeric(data[road_width_col], errors="coerce")
        data["road_width_upto_9m"] = np.where(road_width < 9, 1, 0)
        data["road_width_9_to_18m"] = np.where((road_width >= 9) & (road_width <= 18), 1, 0)
        data["road_width_18_plus"] = np.where(road_width > 18, 1, 0)
        return data

    @staticmethod
    def _make_ohe():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _knn_impute_pass_1(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["facing_direction"] = df["facing_direction"].replace(r"^\s*$", np.nan, regex=True)

        knn_features = [
            "usage_type",
            "locality",
            "city",
            "latitude",
            "longitude",
            "plot_area",
            "road_width_m",
            "property_type",
            "allowed_floors",
            "is_park_facing",
            "is_corner",
            "is_rectangular",
            "is_gated",
            "has_backlane",
            "has_boundary_wall",
        ]

        available_features = [c for c in knn_features if c in df.columns]
        numeric_features = [c for c in available_features if pd.api.types.is_numeric_dtype(df[c])]
        categorical_features = [c for c in available_features if c not in numeric_features]

        facing_known = df["facing_direction"].notna()
        facing_null = df["facing_direction"].isna()

        if facing_null.sum() == 0 or facing_known.sum() < 5:
            return df

        if df.loc[facing_known, "facing_direction"].nunique() < 2:
            df.loc[facing_null, "facing_direction"] = df.loc[facing_known, "facing_direction"].mode().iat[0]
            return df

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="median")),
                            ("scl", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            ("enc", self._make_ohe()),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

        knn_model = Pipeline(
            [
                ("pre", preprocessor),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=min(7, int(facing_known.sum())),
                        weights="distance",
                    ),
                ),
            ]
        )

        knn_model.fit(
            df.loc[facing_known, available_features],
            df.loc[facing_known, "facing_direction"],
        )
        df.loc[facing_null, "facing_direction"] = knn_model.predict(df.loc[facing_null, available_features])
        return df

    @staticmethod
    def _build_preprocessor(df: pd.DataFrame, feature_cols: list[str]):
        num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in feature_cols if c not in num_cols]
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="median")),
                            ("scl", StandardScaler()),
                        ]
                    ),
                    num_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            ("enc", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_cols,
                ),
            ]
        )

    def _knn_impute_pass_2(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["road_width_m"] = pd.to_numeric(df["road_width_m"], errors="coerce")
        df["facing_direction"] = df["facing_direction"].replace(r"^\s*$", np.nan, regex=True)

        if "allowed_floors" in df.columns:
            df["allowed_floors_num"] = pd.to_numeric(df["allowed_floors"], errors="coerce")
        else:
            df["allowed_floors_num"] = np.nan

        base_feature_cols = [
            c
            for c in [
                "usage_type",
                "locality",
                "city",
                "latitude",
                "longitude",
                "allowed_floors_num",
                "is_park_facing",
                "plot_area",
                "total_price",
                "property_type",
                "is_corner",
                "is_rectangular",
                "is_gated",
                "has_backlane",
                "has_boundary_wall",
                "price_per_sqft",
            ]
            if c in df.columns
        ]

        road_feature_cols = [c for c in base_feature_cols + ["facing_direction"] if c in df.columns]
        road_known = df["road_width_m"].notna()
        road_missing = df["road_width_m"].isna()
        if road_missing.any() and road_known.sum() > 1:
            road_prep = self._build_preprocessor(df, road_feature_cols)
            x_train = road_prep.fit_transform(df.loc[road_known, road_feature_cols])
            x_pred = road_prep.transform(df.loc[road_missing, road_feature_cols])
            road_knn = KNeighborsRegressor(
                n_neighbors=min(7, int(road_known.sum())),
                weights="distance",
            )
            road_knn.fit(x_train, df.loc[road_known, "road_width_m"])
            df.loc[road_missing, "road_width_m"] = road_knn.predict(x_pred).round(2)

        facing_feature_cols = [c for c in base_feature_cols + ["road_width_m"] if c in df.columns]
        facing_known = df["facing_direction"].notna()
        facing_missing = df["facing_direction"].isna()
        if facing_missing.any() and facing_known.sum() > 1:
            n_classes = df.loc[facing_known, "facing_direction"].nunique()
            if n_classes == 1:
                df.loc[facing_missing, "facing_direction"] = df.loc[facing_known, "facing_direction"].mode().iat[0]
            else:
                facing_prep = self._build_preprocessor(df, facing_feature_cols)
                x_train = facing_prep.fit_transform(df.loc[facing_known, facing_feature_cols])
                x_pred = facing_prep.transform(df.loc[facing_missing, facing_feature_cols])
                facing_knn = KNeighborsClassifier(
                    n_neighbors=min(7, int(facing_known.sum())),
                    weights="distance",
                )
                facing_knn.fit(x_train, df.loc[facing_known, "facing_direction"].astype(str))
                df.loc[facing_missing, "facing_direction"] = facing_knn.predict(x_pred)

        return df

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
        Fill missing circle_rate using nearest known row by lat/lon (haversine distance).

        Matching priority for donor pool:
          1) Same city + same mapped property type
          2) Same city
          3) Same mapped property type
          4) Any row with known circle_rate
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
            donor_mask = known_mask & (df["_circle_city_key"] == city_key) & (df["_circle_prop_type"] == prop_type)
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
            df.loc[target_index, "circle_rate"] = df.loc[nearest_index, "circle_rate"].to_numpy()
            df.loc[target_index, "_circle_rate_fill_dist_km"] = dist_rad[:, 0] * earth_radius_km
            filled_rows += len(target_index)

        remaining_missing = int((df["circle_rate"].isna() & valid_coord_mask).sum())
        if filled_rows > 0 and "_circle_rate_fill_dist_km" in df.columns:
            median_fill_distance_km = float(df["_circle_rate_fill_dist_km"].median())
            logging.info(
                "Haversine circle-rate fallback filled rows=%s, remaining_missing_with_coords=%s, median_distance_km=%.2f",
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

    def _add_circle_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.circle_rate_matcher is None:
            df["circle_rate"] = np.nan
            return df

        def lookup(row):
            city = row.get("city", "")
            locality = row.get("locality", "")
            prop_type = self._map_prop_type_for_circle(row.get("property_type", ""))
            return self.circle_rate_matcher.get_rate(city, locality, prop_type=prop_type)

        df["circle_rate"] = df.apply(lookup, axis=1)
        direct_matched = int(df["circle_rate"].notna().sum())

        df = self._fill_circle_rate_by_nearest_haversine(df)
        final_matched = int(df["circle_rate"].notna().sum())

        logging.info(
            f"Plot circle-rate matched rows direct={direct_matched}/{len(df)}, after_haversine_fill={final_matched}/{len(df)}"
        )
        return df

    def _add_road_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        class_to_col = {
            "MDR": "closest_distance_MDR_km",
            "SH": "closest_distance_SH_km",
            "NH": "closest_distance_NH_km",
        }
        for col in class_to_col.values():
            if col not in df.columns:
                df[col] = np.nan

        available_geojsons = [p for p in ROADS_GEOJSON_PATHS if os.path.exists(p)]
        if not available_geojsons:
            logging.warning(f"No road geojson files found. Skipping road distance features.")
            return df

        try:
            gpd = importlib.import_module("geopandas")
        except Exception:
            logging.warning("geopandas unavailable. Skipping road distance features.")
            return df

        road_frames = []
        for geojson_path in available_geojsons:
            gdf = gpd.read_file(geojson_path)
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            road_frames.append(gdf)
        import pandas as _pd
        roads = _pd.concat(road_frames, ignore_index=True) if len(road_frames) > 1 else road_frames[0]
        if not isinstance(roads, gpd.GeoDataFrame):
            roads = gpd.GeoDataFrame(roads, crs=road_frames[0].crs)

        roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
        if "ref" not in roads.columns:
            roads["ref"] = np.nan
        # Normalise dehradun_highways.geojson which uses the OSM-native "highway"
        # key instead of the "fclass" key used by all other files.
        if "fclass" not in roads.columns:
            if "highway" in roads.columns:
                roads["fclass"] = roads["highway"]
            else:
                roads["fclass"] = np.nan
        elif "highway" in roads.columns:
            # Fill fclass gaps from highway for any rows that have highway but not fclass
            roads["fclass"] = roads["fclass"].where(
                roads["fclass"].notna() & (roads["fclass"].astype(str).str.strip() != ""),
                other=roads["highway"],
            )
        roads["road_class"] = roads.apply(
            lambda row: self._classify_road_class(row["ref"], row.get("fclass")), axis=1
        )

        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        valid_mask = df["latitude"].notna() & df["longitude"].notna()

        points = gpd.GeoDataFrame(
            df.loc[valid_mask].copy(),
            geometry=gpd.points_from_xy(df.loc[valid_mask, "longitude"], df.loc[valid_mask, "latitude"]),
            crs="EPSG:4326",
        )

        roads_m = roads.to_crs(epsg=3857)
        points_m = points.to_crs(epsg=3857)

        for road_class, out_col in class_to_col.items():
            roads_subset = roads_m[roads_m["road_class"] == road_class]
            dist_km = self._nearest_distance_km(points_m, roads_subset, gpd)
            df.loc[dist_km.index, out_col] = dist_km.round(3)

        logging.info(f"Road features loaded rows={len(roads)}")
        return df

    def _apply_core_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        extracted_df = pd.DataFrame(df["description"].apply(self._extract_des_features).tolist())
        df = pd.concat([df, extracted_df], axis=1)

        df["des_amenities"] = df["description"].apply(self._extract_des_amenities)

        df["covered_area_sqft"] = df.apply(
            lambda r: self._convert_to_sqft(r.get("covered_area_value"), r.get("covered_area_unit")),
            axis=1,
        )
        df["des_covered_area_sqft"] = df.apply(
            lambda r: self._convert_to_sqft(r.get("des_covered_area_value"), r.get("des_covered_area_unit")),
            axis=1,
        )

        if "facing_direction" in df.columns:
            df["facing_direction"] = df["facing_direction"].apply(self._standardize_facing)

        df["final_area"] = df.apply(
            lambda r: self._merge_numeric(r.get("covered_area_sqft"), r.get("des_covered_area_sqft")),
            axis=1,
        )
        df["final_price"] = df.apply(
            lambda r: self._merge_numeric(r.get("price_numeric"), r.get("des_price_numeric")),
            axis=1,
        )
        df["final_road_width"] = df.apply(
            lambda r: self._merge_numeric(r.get("road_width"), r.get("des_road_width")),
            axis=1,
        )

        df["final_facing"] = df.apply(
            lambda r: self._merge_categorical(r.get("facing_direction"), r.get("des_facing_direction")),
            axis=1,
        )
        df["final_property_type"] = df.apply(
            lambda r: self._merge_categorical(r.get("property_type"), r.get("des_property_type")),
            axis=1,
        )

        for struct_col, des_col in [
            ("corner_property", "des_corner_property"),
            ("rectangular_plot", "des_rectangular_plot"),
            ("gated_plot", "des_gated_plot"),
            ("backlane", "des_backlane"),
        ]:
            out_col = f"final_{struct_col}"
            df[out_col] = df.apply(lambda r, c1=struct_col, c2=des_col: self._merge_bool(r.get(c1), r.get(c2)), axis=1)

        df["final_amenities"] = df.apply(
            lambda r: self._merge_amenities(r.get("amenities"), r.get("des_amenities")),
            axis=1,
        )

        df["final_boundary_wall"] = df["final_amenities"].apply(
            lambda v: 1 if self._has_kw(v, "boundary_wall") else 0
        )

        if "final_gated_plot" in df.columns:
            mask = (df["final_gated_plot"] == 0) & df["final_amenities"].apply(
                lambda v: self._has_kw(v, "gated_community")
            )
            df.loc[mask, "final_gated_plot"] = 1

        if "des_park_facing" in df.columns:
            mask = (df["des_park_facing"] == 0) & df["final_amenities"].apply(
                lambda v: self._has_kw(v, "park")
            )
            df.loc[mask, "des_park_facing"] = 1

        rename_dict = {
            "property_usage": "usage_type",
            "address_full": "full_address",
            "des_allowed_floors": "allowed_floors",
            "des_park_facing": "is_park_facing",
            "final_area": "plot_area",
            "final_price": "total_price",
            "final_road_width": "road_width",
            "final_facing": "facing_direction",
            "final_property_type": "property_type",
            "final_corner_property": "is_corner",
            "final_rectangular_plot": "is_rectangular",
            "final_gated_plot": "is_gated",
            "final_backlane": "has_backlane",
            "final_amenities": "amenities",
            "final_boundary_wall": "has_boundary_wall",
        }
        df = df.rename(columns=rename_dict)

        # Keep the derived final columns when source and final names collide.
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="last")]

        df["road_width"] = df.get("road_width", np.nan).fillna("No Info")
        df["road_width_m"] = df["road_width"].apply(self._parse_road_width_m)

        if "total_price" in df.columns:
            df = df.dropna(subset=["total_price"])

        df["price_per_sqft"] = df["total_price"] / df["plot_area"].replace(0, np.nan)

        df = df[df["plot_area"].between(300, 100_000)].reset_index(drop=True)
        df = df[df["price_per_sqft"].between(1000, 350000)].reset_index(drop=True)

        return df

    def _finalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        drop_cols = [
            "amenities",
            "allowed_floors_num",
            "allowed_floors",
            "source",
            "description",
            "full_address",
            "city",
            "road_width",
            "road_width_m",
            "has_backlane",
            "property_type",
            "covered_area_value",
            "covered_area_unit",
            "price_numeric",
            "des_covered_area_value",
            "des_covered_area_unit",
            "des_price_numeric",
            "des_property_type",
            "des_facing_direction",
            "des_corner_property",
            "des_rectangular_plot",
            "des_road_width",
            "des_gated_plot",
            "des_backlane",
            "des_property_status",
            "des_tenure_type",
            "des_rera_approved",
            "covered_area_sqft",
            "des_covered_area_sqft",
            "des_amenities",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        if "latitude" in df.columns and "longitude" in df.columns:
            df = df.dropna(subset=["latitude", "longitude"])

        final_columns = [
            "latitude",
            "longitude",
            "is_park_facing",
            "plot_area",
            "total_price",
            "is_corner",
            "is_rectangular",
            "is_gated",
            "has_boundary_wall",
            "price_per_sqft",
            "closest_distance_MDR_km",
            "closest_distance_SH_km",
            "closest_distance_NH_km",
            "circle_rate",
            "road_width_upto_9m",
            "road_width_9_to_18m",
            "road_width_18_plus",
            "locality",
            "usage_type",
            "facing_direction",
            "price_by_circle_ratio",
        ]

        for col in final_columns:
            if col not in df.columns:
                df[col] = np.nan

        return df[final_columns].reset_index(drop=True)

    def initiate_data_transformation(self) -> PlotDataTransformationArtifact:
        try:
            logging.info("PlotDataTransformation started")

            df = pd.read_csv(self.data_ingestion_artifact.merged_file_path, low_memory=False)
            logging.info(f"Loaded plot merged file shape={df.shape}")

            if "description" not in df.columns:
                df["description"] = ""

            df = self._apply_core_transforms(df)
            logging.info(f"After core transforms shape={df.shape}")

            df = self._knn_impute_pass_1(df)
            df = self._knn_impute_pass_2(df)
            logging.info(
                f"After KNN imputation road_width_m_nulls={df['road_width_m'].isna().sum()} "
                f"facing_nulls={df['facing_direction'].isna().sum()}"
            )

            df = self._add_circle_rate(df)
            df = self._add_road_distance_features(df)
            df = self._add_road_width_flags(df, road_width_col="road_width_m")

            df["price_by_circle_ratio"] = np.where(
                (pd.to_numeric(df.get("circle_rate"), errors="coerce") > 0),
                df["price_per_sqft"] / df["circle_rate"],
                np.nan,
            )

            df = self._finalize_schema(df)

            os.makedirs(self.config.transformed_data_dir, exist_ok=True)
            output_path = self.config.transformed_file_path
            df.to_csv(output_path, index=False)

            logging.info(f"Plot transformed data saved -> {output_path} shape={df.shape}")
            return PlotDataTransformationArtifact(transformed_file_path=output_path)

        except Exception as e:
            raise RealEstateException(e, sys)
