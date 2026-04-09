# ═══════════════════════════════════════════════════════════════
#  NCR Real Estate Price Estimator
#  Models: Apartment/Flat  |  Builder Floor  |  Plot/Land
#  Features: locality → lat/lon (geocode) + circle rate (JSON lookup)
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import glob
import os
import re
from difflib import get_close_matches
from rapidfuzz import fuzz

# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="NCR Real Estate Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; padding: 0 24px;
        font-size: 16px; font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] { background-color: #1f4e79; color: white; }
    .price-card {
        background: linear-gradient(135deg, #1f4e79 0%, #2e86de 100%);
        border-radius: 16px; padding: 28px 32px;
        color: white; text-align: center; margin-top: 20px;
    }
    .price-label { font-size: 14px; opacity: 0.85; text-transform: uppercase; letter-spacing: 1px; }
    .price-value { font-size: 42px; font-weight: 800; margin: 8px 0 4px; }
    .price-sub   { font-size: 15px; opacity: 0.75; }
    .metric-row  { display: flex; gap: 16px; margin-top: 16px; flex-wrap: wrap; }
    .metric-box  {
        flex: 1; background: rgba(255,255,255,0.15);
        border-radius: 10px; padding: 14px; text-align: center; min-width: 150px;
    }
    .metric-box-green {
        background: rgba(46, 139, 87, 0.40);
        border: 1px solid rgba(181, 240, 201, 0.65);
    }
    .metric-box .lbl { font-size: 11px; opacity: 0.8; }
    .metric-box .val { font-size: 20px; font-weight: 700; }
    .info-chip {
        display: inline-block; background: #e8f4fd;
        border-radius: 20px; padding: 4px 12px;
        font-size: 13px; color: #1f4e79; margin: 4px 2px; border: 1px solid #b8d9f5;
    }
    .section-header {
        font-size: 15px; font-weight: 700; color: #1f4e79;
        text-transform: uppercase; letter-spacing: 0.5px;
        border-bottom: 2px solid #2e86de; padding-bottom: 4px; margin: 20px 0 14px;
    }
    div[data-testid="stNumberInput"] label { font-weight: 600; }
    div[data-testid="stSelectbox"] label { font-weight: 600; }
    .warning-box {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 12px 16px; border-radius: 6px; font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING — Circle Rates
# ═══════════════════════════════════════════════════════════════
CR_FOLDER = "real_estate_data/circle_rates"
RENTAL_YIELD_FILE = "artifact/rent_transformation/locality_rental_yield.csv"
PLOT_TRANSFORMED_FILE = "artifact/data_transformation/cleaned_data/plot_transformed.csv"


def _norm_text(value: str) -> str:
    """Normalize text for robust key matching."""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _normalize_city_key(city: str) -> str:
    c = _norm_text(city)
    city_map = {
        "new delhi": "delhi",
        "delhi": "delhi",
        "gurugram": "gurgaon",
        "gurgaon": "gurgaon",
        "greater noida": "greater noida",
        "greater noida west": "greater noida",
        "gr noida": "greater noida",
        "noida": "noida",
        "ghaziabad": "ghaziabad",
        "faridabad": "faridabad",
    }
    return city_map.get(c, c)


def _city_from_filename(filename: str) -> str | None:
    f = _norm_text(filename)
    if "merged_delhi_localities" in f or f.startswith("delhi"):
        return "delhi"
    if "greater-noida" in f or "greater_noida" in f:
        return "greater noida"
    if "noida_circle_rate" in f:
        return "noida"
    if "gurgaon" in f or "gurugram" in f:
        return "gurgaon"
    if "ghaziabad" in f:
        return "ghaziabad"
    if "faridabad" in f:
        return "faridabad"
    return None


def _city_rate_candidates(city: str) -> list[str]:
    city_key = _normalize_city_key(city)
    if city_key == "greater noida":
        return ["greater noida", "noida"]
    return [city_key]


def _get_city_circle_rates(city: str, circle_rates_by_city: dict) -> dict:
    merged: dict[str, float] = {}
    for c in _city_rate_candidates(city):
        bucket = circle_rates_by_city.get(c, {})
        for loc, rate in bucket.items():
            merged.setdefault(loc, rate)
    return merged


_SECTOR_TOKEN_RE = re.compile(r"\bsec(?:tor)?[.\-\s]*(\d{1,3}[a-z]?)\b", re.IGNORECASE)
_LOCALITY_STOPWORDS = {
    "sector",
    "sec",
    "block",
    "phase",
    "extension",
    "extn",
    "ext",
}


def _extract_sector_token(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    m = _SECTOR_TOKEN_RE.search(_norm_text(value))
    return m.group(1).lower() if m else None


def _tokenize_locality(value: str) -> set[str]:
    text = _norm_text(value)
    words = re.findall(r"[a-z0-9]+", text)
    return {w for w in words if w not in _LOCALITY_STOPWORDS}


def _locality_match_score(query: str, candidate: str) -> float:
    """Score locality similarity with sector-aware, token-aware ranking."""
    q = _norm_text(query)
    c = _norm_text(candidate)
    if not q or not c:
        return 0.0

    if q == c:
        return 1000.0

    # Base lexical similarity from rapidfuzz.
    score = max(
        fuzz.token_set_ratio(q, c),
        fuzz.token_sort_ratio(q, c),
        fuzz.partial_ratio(q, c),
    )

    q_tokens = _tokenize_locality(q)
    c_tokens = _tokenize_locality(c)
    if q_tokens:
        overlap_ratio = len(q_tokens & c_tokens) / len(q_tokens)
        score += overlap_ratio * 28.0
        if q_tokens.issubset(c_tokens):
            score += 10.0

    q_sector = _extract_sector_token(q)
    c_sector = _extract_sector_token(c)
    if q_sector and c_sector:
        if q_sector == c_sector:
            score += 35.0
        else:
            score -= 25.0
    elif q_sector and not c_sector:
        score -= 10.0

    if c.startswith(q):
        score += 12.0
    elif q in c:
        score += 6.0

    return score


def _best_locality_match(query: str, candidates: list[str]) -> tuple[str | None, float]:
    q = _norm_text(query)
    if not q or not candidates:
        return (None, 0.0)

    best_key = None
    best_score = 0.0
    for cand in candidates:
        s = _locality_match_score(q, cand)
        if s > best_score:
            best_score = s
            best_key = cand
    return (best_key, best_score)

@st.cache_data(show_spinner=False)
def load_all_circle_rates() -> dict:
    """
    Loads all circle rate JSON files as city buckets:
        { city_key: { locality_key: rate_inr_per_sqft } }
    Handles all circle-rate formats in the circle_rates folder.
    """
    by_city: dict[str, dict[str, float]] = {}

    def _put(city_key: str, locality: str, rate: float) -> None:
        c = _normalize_city_key(city_key)
        loc = _norm_text(locality)
        if not c or not loc:
            return
        by_city.setdefault(c, {})[loc] = float(rate)

    for fpath in glob.glob(os.path.join(CR_FOLDER, "*.json")):
        fname = os.path.basename(fpath).lower()
        file_city = _city_from_filename(fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # ── Format A: list of {locality, circle_land_cost_inr_per_sqft, ...}
        if isinstance(data, list):
            if file_city is None:
                continue
            for item in data:
                if isinstance(item, dict):
                    loc = str(item.get("locality", "")).strip()
                    rate = (
                        item.get("circle_land_cost_inr_per_sqft")
                        if item.get("circle_land_cost_inr_per_sqft") is not None
                        else item.get("rate_2025_per_sqft")
                    )
                    if rate is None:
                        rate = item.get("circle_rate_sqft")
                    if rate is None:
                        rate = item.get("rate")

                    if loc and rate is not None:
                        try:
                            _put(file_city, loc, float(rate))
                        except (TypeError, ValueError):
                            continue

        elif isinstance(data, dict):
            for key, val in data.items():
                key_s = str(key).strip()

                # ── Format B: {locality: rate_float}  (Noida, Faridabad, Ghaziabad)
                if isinstance(val, (int, float)):
                    if file_city is not None:
                        _put(file_city, key_s, float(val))

                # ── Format C: {locality: [{property_sub_type, circle_rate_sqft}]}  (Gurgaon)
                elif isinstance(val, list) and val and isinstance(val[0], dict):
                    if file_city is None:
                        continue
                    # take residential rate; fallback to first entry
                    chosen = None
                    for entry in val:
                        sub = str(entry.get("property_sub_type", "")).lower()
                        if "नि" in sub or "res" in sub or chosen is None:
                            chosen = entry.get("circle_rate_sqft")
                    if chosen is not None:
                        _put(file_city, key_s, float(chosen))

                # ── Format D: {city: {locality: rate | None}}  (missing_circle_rates)
                elif isinstance(val, dict):
                    for sub_loc, sub_val in val.items():
                        if sub_val is not None:
                            try:
                                rate_val = float(sub_val)
                            except (TypeError, ValueError):
                                continue
                            _put(key_s, str(sub_loc), rate_val)

    return by_city


def lookup_circle_rate(locality: str, city: str, circle_rates_by_city: dict) -> float | None:
    """Exact then sector-aware fuzzy match of locality → circle rate (city-filtered)."""
    if not locality:
        return None

    circle_rates = _get_city_circle_rates(city, circle_rates_by_city)
    if not circle_rates:
        return None

    key = _norm_text(locality)
    if key in circle_rates:
        return circle_rates[key]

    best_key, best_score = _best_locality_match(key, list(circle_rates.keys()))
    if best_key is not None and best_score >= 72.0:
        return circle_rates[best_key]

    return None


def fuzzy_locality_suggestions(
    query: str, city: str, circle_rates_by_city: dict, n: int = 6
) -> list[str]:
    """Return top-N locality suggestions from selected city's circle-rate dict."""
    q = _norm_text(query)
    if not q:
        return []

    circle_rates = _get_city_circle_rates(city, circle_rates_by_city)
    if not circle_rates:
        return []

    scored: list[tuple[float, str]] = []
    for cand in circle_rates.keys():
        s = _locality_match_score(q, cand)
        if s >= 40.0:
            scored.append((s, cand))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [cand.title() for _, cand in scored[:n]]


@st.cache_data(show_spinner=False)
def load_rental_yield_table() -> pd.DataFrame:
    """Load locality rental yield lookup table. Returns empty DataFrame if missing."""
    if not os.path.exists(RENTAL_YIELD_FILE):
        return pd.DataFrame(columns=["city_key", "locality_key", "rental_yield_pct", "total_properties"])

    try:
        df = pd.read_csv(RENTAL_YIELD_FILE)
    except Exception:
        return pd.DataFrame(columns=["city_key", "locality_key", "rental_yield_pct", "total_properties"])

    required = {"city", "locality", "rental_yield_pct"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["city_key", "locality_key", "rental_yield_pct", "total_properties"])

    df = df.copy()
    df["city_key"] = df["city"].apply(_norm_text)
    df["locality_key"] = df["locality"].apply(_norm_text)
    df["rental_yield_pct"] = pd.to_numeric(df["rental_yield_pct"], errors="coerce")
    if "total_properties" not in df.columns:
        df["total_properties"] = 0
    df["total_properties"] = pd.to_numeric(df["total_properties"], errors="coerce").fillna(0)

    return df.dropna(subset=["city_key", "locality_key", "rental_yield_pct"])


def lookup_rental_yield(locality: str, city: str, rental_yield_df: pd.DataFrame) -> float | None:
    """Lookup rental yield by city+locality. Falls back to locality-only best-supported row."""
    if not locality or rental_yield_df.empty:
        return None

    loc_key = _norm_text(locality)
    city_key = _norm_text(city)

    city_candidates = {city_key}
    if city_key == "delhi":
        city_candidates.add("new delhi")
    elif city_key == "new delhi":
        city_candidates.add("delhi")

    city_loc_rows = rental_yield_df[
        (rental_yield_df["locality_key"] == loc_key)
        & (rental_yield_df["city_key"].isin(city_candidates))
    ]
    if not city_loc_rows.empty:
        picked = city_loc_rows.sort_values("total_properties", ascending=False).iloc[0]
        return float(picked["rental_yield_pct"])

    locality_rows = rental_yield_df[rental_yield_df["locality_key"] == loc_key]
    if locality_rows.empty:
        return None

    picked = locality_rows.sort_values("total_properties", ascending=False).iloc[0]
    return float(picked["rental_yield_pct"])


@st.cache_data(show_spinner=False)
def load_plot_locality_encoding() -> dict:
    """Load locality target encoding map for Plot model fallback features."""
    empty = {"map": {}, "default": 0.0}
    if not os.path.exists(PLOT_TRANSFORMED_FILE):
        return empty

    try:
        df = pd.read_csv(PLOT_TRANSFORMED_FILE)
    except Exception:
        return empty

    required = {"locality", "locality_target_encoding"}
    if not required.issubset(df.columns):
        return empty

    df = df.copy()
    df["locality_key"] = df["locality"].apply(_norm_text)
    df["locality_target_encoding"] = pd.to_numeric(df["locality_target_encoding"], errors="coerce")
    df = df.dropna(subset=["locality_key", "locality_target_encoding"])
    if df.empty:
        return empty

    enc_map = df.groupby("locality_key")["locality_target_encoding"].mean().to_dict()
    default_val = float(df["locality_target_encoding"].median())
    return {"map": enc_map, "default": default_val}


def lookup_locality_target_encoding(locality: str, locality_encoding_bundle: dict) -> float:
    """Resolve locality target encoding for Plot model; fallback to global median."""
    enc_map = locality_encoding_bundle.get("map", {})
    default_val = float(locality_encoding_bundle.get("default", 0.0))

    if not locality or not enc_map:
        return default_val

    key = _norm_text(locality)
    if key in enc_map:
        return float(enc_map[key])

    matches = get_close_matches(key, enc_map.keys(), n=1, cutoff=0.70)
    if matches:
        return float(enc_map[matches[0]])
    return default_val


# ═══════════════════════════════════════════════════════════════
#  GEOCODING — locality → (lat, lon)
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def geocode_locality(locality: str, city: str = "Delhi") -> tuple[float, float] | None:
    """Use Nominatim to geocode locality + city → (lat, lon)."""
    try:
        from geopy.geocoders import Nominatim
        import time
        geolocator = Nominatim(user_agent="ncr_real_estate_estimator_v2")
        query = f"{locality}, {city}, India"
        loc = geolocator.geocode(query, timeout=5)
        if loc:
            return (loc.latitude, loc.longitude)
        # fallback without city
        loc = geolocator.geocode(f"{locality}, India", timeout=5)
        return (loc.latitude, loc.longitude) if loc else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════
MODEL_BF_PATH    = "notebooks/sell/bf/best_bf_random_forest.pkl"
MODEL_BF_RENT_PATHS = [
    "notebooks/rent/bf/best_bf_random_forest.pkl",
    # Backward-compatible fallback for previously saved BF rent artifacts.
    "notebooks/rent/bf/best_apt_random_forest.pkl",
]
MODEL_APT_PATH   = "notebooks/sell/apt/best_apt_random_forest.pkl"
MODEL_PLOT_PATH  = "notebooks/sell/plot/best_plot_random_forest.pkl"
MODEL_APT_RENT_PATH = "notebooks/rent/apt/best_apt_random_forest.pkl"
BF_VORONOI_PATH  = "notebooks/sell/bf/bf_vor_kmeans.pkl"
APT_VORONOI_PATH = "notebooks/sell/apt/apt_vor_kmeans.pkl"
PLOT_VORONOI_PATH = "notebooks/sell/plot/plot_vor_kmeans.pkl"

@st.cache_resource
def load_bf_model():
    if not os.path.exists(MODEL_BF_PATH):
        return None
    return joblib.load(MODEL_BF_PATH)

@st.cache_resource
def load_bf_rent_model():
    for path in MODEL_BF_RENT_PATHS:
        if os.path.exists(path):
            return joblib.load(path)
    return None


@st.cache_resource
def load_apt_model():
    if not os.path.exists(MODEL_APT_PATH):
        return None
    return joblib.load(MODEL_APT_PATH)


@st.cache_resource
def load_apt_rent_model():
    if not os.path.exists(MODEL_APT_RENT_PATH):
        return None
    return joblib.load(MODEL_APT_RENT_PATH)


@st.cache_resource
def load_plot_model():
    if not os.path.exists(MODEL_PLOT_PATH):
        return None
    return joblib.load(MODEL_PLOT_PATH)


@st.cache_resource
def load_bf_voronoi():
    if not os.path.exists(BF_VORONOI_PATH):
        return None
    return joblib.load(BF_VORONOI_PATH)


@st.cache_resource
def load_apt_voronoi():
    if not os.path.exists(APT_VORONOI_PATH):
        return None
    return joblib.load(APT_VORONOI_PATH)


@st.cache_resource
def load_plot_voronoi():
    if not os.path.exists(PLOT_VORONOI_PATH):
        return None
    return joblib.load(PLOT_VORONOI_PATH)


# ═══════════════════════════════════════════════════════════════
#  VORONOI FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════
def compute_voronoi_features(lat: float, lon: float, vor_kmeans) -> dict:
    """Compute voronoi_dist_to_seed + vor_cell_0..N for a single point.
    Works for Builder Floor, Apartment, and Plot KMeans models."""
    n_cells = vor_kmeans.n_clusters
    point = np.array([[lat, lon]])
    cell_id = int(vor_kmeans.predict(point)[0])
    centers = vor_kmeans.cluster_centers_
    dist = float(np.sqrt(((point - centers[cell_id]) ** 2).sum()))

    feats = {"voronoi_dist_to_seed": dist}
    for i in range(n_cells):
        feats[f"vor_cell_{i}"] = 1 if i == cell_id else 0
    return feats


# ═══════════════════════════════════════════════════════════════
#  FEATURE VECTOR BUILDERS
# ═══════════════════════════════════════════════════════════════
AGE_CATEGORIES        = ["10 to 20 years", "5 to 10 years", "Above 20 years",
                          "Less than 5 years", "New Construction"]
FURNISHING_CATEGORIES = ["Furnished", "Semi-Furnished", "Unfurnished"]
FACING_CATEGORIES     = ["East", "North", "North-East", "North-West",
                          "South", "South-East", "South-West", "West"]


def _ohe(value: str, categories: list, prefix: str) -> dict:
    return {f"{prefix}_{c}": 1 if c == value else 0 for c in categories}


def build_features(
    *,
    bhk: int, area_sqft: float, bathrooms: int, balconies: int,
    circle_rate: float, is_parking: int, is_pool: int,
    is_main_road: int, is_garden_park: int, is_gated: int, is_corner: int,
    age: str, furnishing: str, facing: str,
    voronoi_feats: dict,
    model,
    floor_feats: dict | None = None,
) -> pd.DataFrame:
    """
    Build the exact feature DataFrame the model expects.
    Uses model.feature_names_in_ to align columns (fills missing with 0).
    """
    base = {
        "bhk": bhk,
        "covered_area_sqft": area_sqft,
        "bathrooms": bathrooms,
        "balconies": balconies,
        "circle_rate": circle_rate,
        "is_parking": is_parking,
        "is_pool": is_pool,
        "is_main_road": is_main_road,
        "is_garden_park": is_garden_park,
        "is_gated": is_gated,
        "is_corner": is_corner,
    }
    base.update(_ohe(age,        AGE_CATEGORIES,        "age"))
    base.update(_ohe(furnishing, FURNISHING_CATEGORIES, "furn"))
    base.update(_ohe(facing,     FACING_CATEGORIES,     "facing"))
    base.update(voronoi_feats)
    if floor_feats:
        base.update(floor_feats)

    # Align to model's expected columns
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        row = {c: base.get(c, 0) for c in cols}
        return pd.DataFrame([row], columns=cols)

    return pd.DataFrame([base])


def build_apartment_property_features(property_segment: str, model) -> dict:
    """Map apartment property segment selection to possible model feature columns."""
    segment = (property_segment or "").strip().lower()
    model_cols = set(getattr(model, "feature_names_in_", [])) if model is not None else set()

    feats = {
        # Generic UI-style property type columns.
        "property_type_base": 0,
        "property_type_mid": 0,
        "property_type_high": 0,
        "property_type_luxury": 0,
        "property_tier_base": 0,
        "property_tier_mid": 0,
        "property_tier_high": 0,
        "property_tier_luxury": 0,
        # Training-time locality tier variants seen across apartment pipelines.
        "locality_tier_budget": 0,
        "locality_tier_mid": 0,
        "locality_tier_high": 0,
        "locality_tier_premium": 0,
        "locality_tier_luxury": 0,
        "locality_tier_ord": 0,
    }

    if segment == "base":
        feats["property_type_base"] = 1
        feats["property_tier_base"] = 1
        feats["locality_tier_budget"] = 1
        feats["locality_tier_ord"] = 0
    elif segment == "mid":
        feats["property_type_mid"] = 1
        feats["property_tier_mid"] = 1
        feats["locality_tier_mid"] = 1
        feats["locality_tier_ord"] = 1
    elif segment == "high":
        feats["property_type_high"] = 1
        feats["property_tier_high"] = 1
        feats["locality_tier_ord"] = 2
        if "locality_tier_high" in model_cols:
            feats["locality_tier_high"] = 1
        elif "locality_tier_premium" in model_cols:
            feats["locality_tier_premium"] = 1
    elif segment == "luxury":
        feats["property_type_luxury"] = 1
        feats["property_tier_luxury"] = 1
        feats["locality_tier_luxury"] = 1
        feats["locality_tier_ord"] = 3
    else:
        # Safe fallback to mid segment if unexpected label is provided.
        feats["property_type_mid"] = 1
        feats["property_tier_mid"] = 1
        feats["locality_tier_mid"] = 1
        feats["locality_tier_ord"] = 1

    return feats


def build_plot_features(
    *,
    area_sqft: float,
    circle_rate: float,
    is_main_road: int,
    is_garden_park: int,
    gated_plot: int,
    corner_property: int,
    age: str,
    facing: str,
    property_type: str,
    locality_target_encoding: float,
    voronoi_feats: dict,
    model,
) -> pd.DataFrame:
    """Build aligned feature vector for Plot model with alias-safe field handling."""
    base = {
        "covered_area_sqft": area_sqft,
        "covered_area": area_sqft,
        "area_sqft": area_sqft,
        "plot_area_sqft": area_sqft,
        "circle_rate": circle_rate,
        "is_main_road": is_main_road,
        "is_garden_park": is_garden_park,
        "gated_plot": gated_plot,
        "corner_property": corner_property,
        "is_gated": gated_plot,
        "is_corner": corner_property,
        "locality_target_encoding": locality_target_encoding,
        "property_type_Residential": 1 if property_type == "Residential" else 0,
        "property_type_Commercial": 1 if property_type == "Commercial" else 0,
    }
    base.update(_ohe(age, AGE_CATEGORIES, "age"))
    base.update(_ohe(facing, FACING_CATEGORIES, "facing"))
    base.update(voronoi_feats)

    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        row = {c: base.get(c, 0) for c in cols}
        return pd.DataFrame([row], columns=cols)

    return pd.DataFrame([base])


# ═══════════════════════════════════════════════════════════════
#  PRICE FORMATTING
# ═══════════════════════════════════════════════════════════════
def fmt_price(p: float) -> str:
    if p >= 1e7:
        return f"₹{p / 1e7:.2f} Cr"
    elif p >= 1e5:
        return f"₹{p / 1e5:.2f} L"
    return f"₹{p:,.0f}"


def fmt_indian_number(value: float) -> str:
    """Format number with Indian digit grouping, e.g. 11123102 -> 1,11,23,102."""
    n = int(round(value))
    sign = "-" if n < 0 else ""
    s = str(abs(n))
    if len(s) <= 3:
        return f"{sign}{s}"

    last_three = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return f"{sign}{','.join(parts + [last_three])}"


# ═══════════════════════════════════════════════════════════════
#  LOAD ALL RESOURCES
# ═══════════════════════════════════════════════════════════════
circle_rates = load_all_circle_rates()
rental_yield_df = load_rental_yield_table()
bf_model     = load_bf_model()
bf_rent_model = load_bf_rent_model()
apt_model    = load_apt_model()
apt_rent_model = load_apt_rent_model()
plot_model   = load_plot_model()
bf_vor       = load_bf_voronoi()
apt_vor      = load_apt_voronoi()
plot_vor     = load_plot_voronoi()
plot_locality_enc = load_plot_locality_encoding()


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR — Locality Search & Location Auto-Fill
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/real-estate.png", width=72)
    st.title("NCR Property\nEstimator")
    st.markdown("---")

    st.markdown("### 📍 Location")

    CITY_OPTIONS = ["Delhi", "Noida", "Gurgaon", "Faridabad", "Ghaziabad",
                    "Greater Noida"]
    city = st.selectbox("City / Region", CITY_OPTIONS)

    city_circle_rates = _get_city_circle_rates(city, circle_rates)

    locality_query = st.text_input(
        "Locality Name",
        placeholder="e.g. Preet Vihar, Sector 50...",
        help="Type your locality to auto-fill circle rate and coordinates",
    )

    # Suggestions
    suggestions = fuzzy_locality_suggestions(locality_query, city, circle_rates)
    chosen_locality = locality_query
    if suggestions and locality_query:
        chosen = st.selectbox(
            "Matching Localities (select to use)",
            ["— type to search —"] + suggestions,
        )
        if chosen != "— type to search —":
            chosen_locality = chosen

    # Circle rate lookup
    auto_cr = lookup_circle_rate(chosen_locality, city, circle_rates)
    auto_rental_yield = lookup_rental_yield(chosen_locality, city, rental_yield_df)

    if auto_cr:
        st.success(f"✅ Circle Rate  →  ₹{auto_cr:,.2f}/sqft")
    elif chosen_locality:
        st.warning("⚠️ Circle rate not found — enter manually below")

    # ── Auto-geocode whenever locality or city changes ─────────
    _last_key  = st.session_state.get("_last_geocoded_locality", "")
    _last_city = st.session_state.get("_last_geocoded_city", "")
    if chosen_locality and (chosen_locality != _last_key or city != _last_city):
        with st.spinner(f"📡 Fetching coordinates for {chosen_locality} …"):
            lat_lon = geocode_locality(chosen_locality, city)
        st.session_state["_last_geocoded_locality"] = chosen_locality
        st.session_state["_last_geocoded_city"]     = city
        if lat_lon:
            # Write directly into every lat/lon widget key so they update instantly
            for _pfx in ("bf", "apt", "plot"):
                st.session_state[f"{_pfx}_lat"] = lat_lon[0]
                st.session_state[f"{_pfx}_lon"] = lat_lon[1]
            st.session_state["auto_lat"] = lat_lon[0]
            st.session_state["auto_lon"] = lat_lon[1]
            st.success(f"📍 {lat_lon[0]:.5f}, {lat_lon[1]:.5f}")
        else:
            st.error("Geocoding failed — enter coordinates manually")
    elif chosen_locality and "auto_lat" in st.session_state:
        st.info(f"📍 {st.session_state['auto_lat']:.5f}, {st.session_state['auto_lon']:.5f}")

    # ── Push auto circle_rate into widget keys ────────────────
    if auto_cr:
        for _pfx in ("bf", "apt", "plot"):
            st.session_state[f"{_pfx}_cr"] = float(auto_cr)

    st.markdown("---")
    st.markdown(
        f"<small>Circle rates loaded for **{city}**: **{len(city_circle_rates):,}** localities</small>",
        unsafe_allow_html=True,
    )

    # Model status
    st.markdown("### 🤖 Model Status")
    st.markdown(
        f"{'✅' if bf_model else '❌'} **Builder Floor** "
        f"`{'Random Forest' if bf_model else 'Not found'}`"
    )
    st.markdown(
        f"{'✅' if bf_rent_model else '❌'} **Builder Floor Rent (for Yield)** "
        f"`{'Random Forest' if bf_rent_model else 'Not found'}`"
    )
    st.markdown(
        f"{'✅' if apt_model else '❌'} **Apartment** "
        f"`{'Random Forest' if apt_model else 'Not found'}`"
    )
    st.markdown(
        f"{'✅' if apt_rent_model else '❌'} **Apartment Rent (for Yield)** "
        f"`{'Random Forest' if apt_rent_model else 'Not found'}`"
    )
    st.markdown(
        f"{'✅' if plot_model else '❌'} **Plot / Land** "
        f"`{'Random Forest' if plot_model else 'Not found'}`"
    )
    if not bf_vor:
        st.caption("⚠️ bf_vor_kmeans.pkl missing — Voronoi features will be zeroed")
    if not apt_vor:
        st.caption("⚠️ apt_vor_kmeans.pkl missing — Voronoi features will be zeroed")
    if not plot_vor:
        st.caption("⚠️ plot_vor_kmeans.pkl missing — Voronoi features will be zeroed")


# ═══════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='color:#1f4e79;margin-bottom:4px'>🏠 NCR Real Estate Price Estimator</h1>"
    "<p style='color:#555;margin-bottom:0'>Predict property prices for the Delhi NCR region · "
    "Circle rates auto-matched from government data</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

tab_bf, tab_apt, tab_plot = st.tabs(["🏠  Builder Floor", "🏢  Apartment / Flat", "🧱  Plot / Land"])


# ── SHARED INPUT BUILDER ──────────────────────────────────────
def shared_inputs(prefix: str, auto_cr: float | None):
    """
    Renders input widgets common to both models.
    Returns a dict of raw user inputs.
    """
    # ── Seed session-state keys only on first visit (never overwrite user edits)
    if f"{prefix}_lat" not in st.session_state:
        st.session_state[f"{prefix}_lat"] = st.session_state.get("auto_lat", 28.6139)
    if f"{prefix}_lon" not in st.session_state:
        st.session_state[f"{prefix}_lon"] = st.session_state.get("auto_lon", 77.2090)
    if f"{prefix}_cr" not in st.session_state:
        st.session_state[f"{prefix}_cr"] = float(auto_cr or 11891.59)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-header">📐 Size & Rooms</div>', unsafe_allow_html=True)
        bhk        = st.number_input("BHK",        min_value=1, max_value=10, value=3,  key=f"{prefix}_bhk")
        area_sqft  = st.number_input("Area (sqft)", min_value=100, max_value=20000, value=1200, key=f"{prefix}_area")
        bathrooms  = st.number_input("Bathrooms",   min_value=1, max_value=10, value=2,  key=f"{prefix}_bath")
        balconies  = st.number_input("Balconies",   min_value=0, max_value=10, value=1,  key=f"{prefix}_balc")

    with c2:
        st.markdown('<div class="section-header">🏗️ Property Details</div>', unsafe_allow_html=True)
        age        = st.selectbox("Age of Property",  AGE_CATEGORIES,        key=f"{prefix}_age")
        furnishing = st.selectbox("Furnishing Type",  FURNISHING_CATEGORIES, key=f"{prefix}_furn")
        facing     = st.selectbox("Facing Direction", FACING_CATEGORIES,     key=f"{prefix}_face")
        circle_rate = st.number_input(
            "Circle Rate (₹/sqft)",
            min_value=100.0, max_value=100000.0, format="%.2f",
            help="Auto-filled from locality; override if needed",
            key=f"{prefix}_cr",
        )

    with c3:
        st.markdown('<div class="section-header">✨ Amenities</div>', unsafe_allow_html=True)
        is_pool       = int(st.checkbox("Swimming Pool",    value=False, key=f"{prefix}_pool"))
        is_garden     = int(st.checkbox("Garden / Park",    value=True,  key=f"{prefix}_garden"))
        is_main_road  = int(st.checkbox("Main Road Access", value=False, key=f"{prefix}_road"))
        is_gated      = int(st.checkbox("Gated Community",  value=True,  key=f"{prefix}_gated"))
        is_corner     = int(st.checkbox("Corner Property",  value=False, key=f"{prefix}_corner"))
        is_parking    = int(st.checkbox("Parking Available",value=True,  key=f"{prefix}_park"))

    # Read lat/lon silently from session state (set by sidebar geocoder)
    lat = st.session_state.get(f"{prefix}_lat", st.session_state.get("auto_lat", 28.6139))
    lon = st.session_state.get(f"{prefix}_lon", st.session_state.get("auto_lon", 77.2090))

    return dict(
        bhk=bhk, area_sqft=area_sqft, bathrooms=bathrooms, balconies=balconies,
        age=age, furnishing=furnishing, facing=facing, circle_rate=circle_rate,
        is_parking=is_parking, is_pool=is_pool, is_main_road=is_main_road,
        is_garden_park=is_garden, is_gated=is_gated, is_corner=is_corner,
        lat=lat, lon=lon,
    )


PLOT_TYPE_OPTIONS = ["Residential", "Commercial"]


def plot_inputs(prefix: str, auto_cr: float | None):
    """Renders plot/land-specific inputs and returns a raw input dict."""
    if f"{prefix}_lat" not in st.session_state:
        st.session_state[f"{prefix}_lat"] = st.session_state.get("auto_lat", 28.6139)
    if f"{prefix}_lon" not in st.session_state:
        st.session_state[f"{prefix}_lon"] = st.session_state.get("auto_lon", 77.2090)
    if f"{prefix}_cr" not in st.session_state:
        st.session_state[f"{prefix}_cr"] = float(auto_cr or 11891.59)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-header">📐 Plot Size</div>', unsafe_allow_html=True)
        area_sqft = st.number_input("Plot Area (sqft)", min_value=100, max_value=50000, value=1800, key=f"{prefix}_area")

    with c2:
        st.markdown('<div class="section-header">🏗️ Plot Details</div>', unsafe_allow_html=True)
        age = st.selectbox("Age of Property", AGE_CATEGORIES, key=f"{prefix}_age")
        facing = st.selectbox("Facing Direction", FACING_CATEGORIES, key=f"{prefix}_face")
        property_type = st.selectbox("Property Type", PLOT_TYPE_OPTIONS, key=f"{prefix}_ptype")
        circle_rate = st.number_input(
            "Circle Rate (₹/sqft)",
            min_value=100.0,
            max_value=100000.0,
            format="%.2f",
            help="Auto-filled from locality; override if needed",
            key=f"{prefix}_cr",
        )

    with c3:
        st.markdown('<div class="section-header">✨ Plot Amenities</div>', unsafe_allow_html=True)
        is_main_road = int(st.checkbox("Main Road Access", value=False, key=f"{prefix}_road"))
        is_garden_park = int(st.checkbox("Garden / Park Nearby", value=True, key=f"{prefix}_garden"))
        gated_plot = int(st.checkbox("Gated Plot", value=True, key=f"{prefix}_gated"))
        corner_property = int(st.checkbox("Corner Plot", value=False, key=f"{prefix}_corner"))

    lat = st.session_state.get(f"{prefix}_lat", st.session_state.get("auto_lat", 28.6139))
    lon = st.session_state.get(f"{prefix}_lon", st.session_state.get("auto_lon", 77.2090))

    return dict(
        area_sqft=area_sqft,
        age=age,
        facing=facing,
        property_type=property_type,
        circle_rate=circle_rate,
        is_main_road=is_main_road,
        is_garden_park=is_garden_park,
        gated_plot=gated_plot,
        corner_property=corner_property,
        lat=lat,
        lon=lon,
    )


def show_result(
    pred_ratio: float,
    circle_rate: float,
    area_sqft: float,
    model_name: str,
    rental_yield_override: float | None = None,
):
    ppsf   = pred_ratio * circle_rate
    total  = ppsf * area_sqft
    total_indian = fmt_indian_number(total)
    ppsf_indian = fmt_indian_number(ppsf)
    area_indian = fmt_indian_number(area_sqft)
    rental_yield_text = "Na"
    if rental_yield_override is not None:
        rental_yield_text = f"{rental_yield_override:.2f}%"
    elif auto_rental_yield is not None:
        rental_yield_text = f"{auto_rental_yield:.2f}%"

    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">{model_name} · Estimated Price</div>
        <div class="price-value">{fmt_price(total)}</div>
        <div class="price-sub">₹{total_indian}</div>
        <div class="metric-row">
            <div class="metric-box"><div class="lbl">₹/sqft</div><div class="val">₹{ppsf_indian}</div></div>
            <div class="metric-box"><div class="lbl">Area</div><div class="val">{area_indian} sqft</div></div>
            <div class="metric-box metric-box-green"><div class="lbl">Rental Yield</div><div class="val">{rental_yield_text}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_plot_result(pred_ratio: float, circle_rate: float, area_sqft: float, model_name: str):
    """
    Plot model target is price_by_circle = total_price / circle_rate.
    So:
      total_price = pred_ratio * circle_rate
      price_per_sqft = total_price / area_sqft
    """
    total = pred_ratio * circle_rate
    ppsf = total / area_sqft if area_sqft > 0 else 0.0
    total_indian = fmt_indian_number(total)
    ppsf_indian = fmt_indian_number(ppsf)
    area_indian = fmt_indian_number(area_sqft)
    rental_yield_text = "Na"
    if auto_rental_yield is not None:
        rental_yield_text = f"{auto_rental_yield:.2f}%"

    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">{model_name} · Estimated Price</div>
        <div class="price-value">{fmt_price(total)}</div>
        <div class="price-sub">₹{total_indian}</div>
        <div class="metric-row">
            <div class="metric-box"><div class="lbl">₹/sqft</div><div class="val">₹{ppsf_indian}</div></div>
            <div class="metric-box"><div class="lbl">Area</div><div class="val">{area_indian} sqft</div></div>
            <div class="metric-box metric-box-green"><div class="lbl">Rental Yield</div><div class="val">{rental_yield_text}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 1 — BUILDER FLOOR
# ═══════════════════════════════════════════════════════════════
with tab_bf:
    st.markdown("### Builder Floor Property")
    st.caption(
        "Predict price for independent builder floor / villa type properties. "
        "Model: Random Forest (trained on NCR builder floor data)."
    )

    if not bf_model:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>Model not found.</b> Run the <code>model_without_metro2.ipynb</code> notebook to train
        and save <code>best_bf_random_forest.pkl</code> and <code>bf_vor_kmeans.pkl</code>.
        </div>
        """, unsafe_allow_html=True)
    else:
        inputs = shared_inputs("bf", auto_cr)

        # ── Always build feature vector from current inputs ───
        _bf_lat, _bf_lon = inputs["lat"], inputs["lon"]
        if bf_vor is not None:
            _bf_vor_feats = compute_voronoi_features(_bf_lat, _bf_lon, bf_vor)
        else:
            _bf_vor_feats = {"voronoi_dist_to_seed": 0.0}
            _bf_vor_feats.update({f"vor_cell_{i}": 0 for i in range(60)})

        X_bf = build_features(
            bhk=inputs["bhk"], area_sqft=inputs["area_sqft"],
            bathrooms=inputs["bathrooms"], balconies=inputs["balconies"],
            circle_rate=inputs["circle_rate"],
            is_parking=inputs["is_parking"], is_pool=inputs["is_pool"],
            is_main_road=inputs["is_main_road"], is_garden_park=inputs["is_garden_park"],
            is_gated=inputs["is_gated"], is_corner=inputs["is_corner"],
            age=inputs["age"], furnishing=inputs["furnishing"], facing=inputs["facing"],
            voronoi_feats=_bf_vor_feats,
            model=bf_model,
        )

        # ── Predict Button ────────────────────────────────────
        st.markdown("")
        predict_btn = st.button(
            "🔮 Predict Builder Floor Price",
            type="primary", use_container_width=True,
            key="btn_bf",
        )

        if predict_btn:
            if not bf_vor:
                st.warning(
                    "⚠️ `bf_vor_kmeans.pkl` not found — "
                    "Voronoi features zeroed; prediction accuracy may be reduced."
                )

            with st.spinner("Predicting …"):
                pred_log = bf_model.predict(X_bf)[0]
                pred_ratio = float(np.expm1(pred_log))

                # Builder-Floor-only rental yield:
                # (predicted_monthly_rent * 12 / predicted_sell_price) * 100
                bf_rental_yield_pred = None
                if bf_rent_model is not None:
                    X_bf_rent = build_features(
                        bhk=inputs["bhk"], area_sqft=inputs["area_sqft"],
                        bathrooms=inputs["bathrooms"], balconies=inputs["balconies"],
                        circle_rate=inputs["circle_rate"],
                        is_parking=inputs["is_parking"], is_pool=inputs["is_pool"],
                        is_main_road=inputs["is_main_road"], is_garden_park=inputs["is_garden_park"],
                        is_gated=inputs["is_gated"], is_corner=inputs["is_corner"],
                        age=inputs["age"], furnishing=inputs["furnishing"], facing=inputs["facing"],
                        voronoi_feats=_bf_vor_feats,
                        model=bf_rent_model,
                    )
                    bf_rent_log_pred = float(bf_rent_model.predict(X_bf_rent)[0])
                    bf_monthly_rent_pred = float(np.expm1(bf_rent_log_pred))

                    bf_sell_price_pred = float(pred_ratio * inputs["circle_rate"] * inputs["area_sqft"])
                    if bf_sell_price_pred > 0:
                        bf_rental_yield_pred = float((bf_monthly_rent_pred * 12.0 / bf_sell_price_pred) * 100.0)

            show_result(
                pred_ratio,
                inputs["circle_rate"],
                inputs["area_sqft"],
                "Builder Floor",
                rental_yield_override=bf_rental_yield_pred,
            )


# ═══════════════════════════════════════════════════════════════
#  TAB 2 — APARTMENT / FLAT
# ═══════════════════════════════════════════════════════════════
FLOOR_LEVELS = ["Low (Ground – 1st)", "Medium (2nd – 7th)", "High (8th+)"]
APT_PROPERTY_SEGMENTS = ["Base", "Mid", "High", "Luxury"]

with tab_apt:
    st.markdown("### Apartment / Flat")
    st.caption(
        "Predict price for multi-storey apartment or flat. "
        "Model trained on NCR apartment data."
    )

    if not apt_model:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>Model not found.</b> Run the <code>model_without_metro3.ipynb</code> notebook to train
        and save <code>best_apt_random_forest.pkl</code> and <code>apt_vor_kmeans.pkl</code>.
        </div>
        """, unsafe_allow_html=True)
    else:
        inputs_apt = shared_inputs("apt", auto_cr)

        # ── Floor-level inputs (apartment-specific) ───────────
        st.markdown('<div class="section-header">🏢 Floor Details</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns([1.6, 1, 1])
        with fc1:
            floor_level = st.selectbox(
                "Floor Level",
                FLOOR_LEVELS,
                index=1,
                key="apt_floor_level",
                help="Based on NCR-wide quartiles: Low ≤ Q25 (1st floor), High ≥ Q75 (8th floor)",
            )
        with fc2:
            is_ground = int(st.checkbox("Ground Floor", value=False, key="apt_ground"))
        with fc3:
            is_top = int(st.checkbox("Top Floor", value=False, key="apt_top"))
        apt_property_segment = st.selectbox(
            "Property Type",
            APT_PROPERTY_SEGMENTS,
            index=1,
            key="apt_property_segment",
            help="Choose apartment segment used by the model: Base, Mid, High, or Luxury.",
        )

        # ── Always build feature vector from current inputs ───
        _apt_lat, _apt_lon = inputs_apt["lat"], inputs_apt["lon"]
        if apt_vor is not None:
            _apt_vor_feats = compute_voronoi_features(_apt_lat, _apt_lon, apt_vor)
        else:
            _apt_vor_feats = {"voronoi_dist_to_seed": 0.0}
            _apt_vor_feats.update({f"vor_cell_{i}": 0 for i in range(60)})

        _apt_floor_feats = {
            "floor_low":    1 if "Low" in floor_level else 0,
            "floor_medium": 1 if "Medium" in floor_level else 0,
            "floor_high":   1 if "High" in floor_level else 0,
            "is_ground_floor": is_ground,
            "is_top_floor":    is_top,
        }
        _apt_floor_feats.update(build_apartment_property_features(apt_property_segment, apt_model))

        X_apt = build_features(
            bhk=inputs_apt["bhk"], area_sqft=inputs_apt["area_sqft"],
            bathrooms=inputs_apt["bathrooms"], balconies=inputs_apt["balconies"],
            circle_rate=inputs_apt["circle_rate"],
            is_parking=inputs_apt["is_parking"], is_pool=inputs_apt["is_pool"],
            is_main_road=inputs_apt["is_main_road"], is_garden_park=inputs_apt["is_garden_park"],
            is_gated=inputs_apt["is_gated"], is_corner=inputs_apt["is_corner"],
            age=inputs_apt["age"], furnishing=inputs_apt["furnishing"],
            facing=inputs_apt["facing"],
            voronoi_feats=_apt_vor_feats,
            model=apt_model,
            floor_feats=_apt_floor_feats,
        )

        # ── Predict Button ────────────────────────────────────
        st.markdown("")
        predict_btn_apt = st.button(
            "🔮 Predict Apartment Price",
            type="primary", use_container_width=True,
            key="btn_apt",
        )

        if predict_btn_apt:
            if not apt_vor:
                st.warning(
                    "⚠️ `apt_vor_kmeans.pkl` not found — "
                    "Voronoi features zeroed; prediction accuracy may be reduced."
                )

            with st.spinner("Predicting …"):
                pred_log_apt = apt_model.predict(X_apt)[0]
                pred_ratio_apt = float(np.expm1(pred_log_apt))

                # Apartment-only rental yield:
                # (predicted_monthly_rent * 12 / predicted_sell_price) * 100
                rental_yield_pred = None
                if apt_rent_model is not None:
                    X_apt_rent = build_features(
                        bhk=inputs_apt["bhk"], area_sqft=inputs_apt["area_sqft"],
                        bathrooms=inputs_apt["bathrooms"], balconies=inputs_apt["balconies"],
                        circle_rate=inputs_apt["circle_rate"],
                        is_parking=inputs_apt["is_parking"], is_pool=inputs_apt["is_pool"],
                        is_main_road=inputs_apt["is_main_road"], is_garden_park=inputs_apt["is_garden_park"],
                        is_gated=inputs_apt["is_gated"], is_corner=inputs_apt["is_corner"],
                        age=inputs_apt["age"], furnishing=inputs_apt["furnishing"],
                        facing=inputs_apt["facing"],
                        voronoi_feats=_apt_vor_feats,
                        model=apt_rent_model,
                        floor_feats=_apt_floor_feats,
                    )
                    rent_log_pred = float(apt_rent_model.predict(X_apt_rent)[0])
                    monthly_rent_pred = float(np.expm1(rent_log_pred))

                    sell_price_pred = float(pred_ratio_apt * inputs_apt["circle_rate"] * inputs_apt["area_sqft"])
                    if sell_price_pred > 0:
                        rental_yield_pred = float((monthly_rent_pred * 12.0 / sell_price_pred) * 100.0)

            show_result(
                pred_ratio_apt,
                inputs_apt["circle_rate"],
                inputs_apt["area_sqft"],
                "Apartment",
                rental_yield_override=rental_yield_pred,
            )


# ═══════════════════════════════════════════════════════════════
#  TAB 3 — PLOT / LAND
# ═══════════════════════════════════════════════════════════════
with tab_plot:
    st.markdown("### Plot / Land")
    st.caption(
        "Predict price for plot/land properties. "
        "Model: Random Forest trained on price_by_circle with Voronoi spatial features."
    )

    if not plot_model:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>Model not found.</b> Run the <code>plot_model.ipynb</code> notebook and save
        <code>best_plot_random_forest.pkl</code> and <code>plot_vor_kmeans.pkl</code>.
        </div>
        """, unsafe_allow_html=True)
    else:
        inputs_plot = plot_inputs("plot", auto_cr)

        _plot_lat, _plot_lon = inputs_plot["lat"], inputs_plot["lon"]
        if plot_vor is not None:
            _plot_vor_feats = compute_voronoi_features(_plot_lat, _plot_lon, plot_vor)
        else:
            _plot_vor_feats = {"voronoi_dist_to_seed": 0.0}

        plot_locality_te = lookup_locality_target_encoding(chosen_locality, plot_locality_enc)
        st.markdown(
            f"<span class='info-chip'>Locality Target Encoding: {plot_locality_te:.3f}</span>",
            unsafe_allow_html=True,
        )

        X_plot = build_plot_features(
            area_sqft=inputs_plot["area_sqft"],
            circle_rate=inputs_plot["circle_rate"],
            is_main_road=inputs_plot["is_main_road"],
            is_garden_park=inputs_plot["is_garden_park"],
            gated_plot=inputs_plot["gated_plot"],
            corner_property=inputs_plot["corner_property"],
            age=inputs_plot["age"],
            facing=inputs_plot["facing"],
            property_type=inputs_plot["property_type"],
            locality_target_encoding=plot_locality_te,
            voronoi_feats=_plot_vor_feats,
            model=plot_model,
        )

        st.markdown("")
        predict_btn_plot = st.button(
            "🔮 Predict Plot Price",
            type="primary",
            use_container_width=True,
            key="btn_plot",
        )

        if predict_btn_plot:
            if not plot_vor:
                st.warning(
                    "⚠️ `plot_vor_kmeans.pkl` not found — "
                    "Voronoi features zeroed; prediction accuracy may be reduced."
                )

            with st.spinner("Predicting …"):
                pred_ratio_plot = float(plot_model.predict(X_plot)[0])
                pred_ratio_plot = max(pred_ratio_plot, 0.01)

            show_plot_result(pred_ratio_plot, inputs_plot["circle_rate"], inputs_plot["area_sqft"], "Plot / Land")


# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small style='color:#999'>NCR Real Estate Estimator · "
    "Predictions based on ML models trained on MagicBricks & Housing.com data · "
    "For reference only</small></center>",
    unsafe_allow_html=True,
)
