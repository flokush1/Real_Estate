import glob
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_startup_logger = logging.getLogger("ncr_api.startup")

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

# Pre-import all sklearn submodules that joblib.load will need.
# This must happen in the main thread BEFORE ThreadPoolExecutor starts,
# because Python 3.14 _ModuleLock raises DeadlockError when multiple threads
# try to import the same module simultaneously.
import sklearn.ensemble
import sklearn.ensemble._iforest
import sklearn.ensemble._forest
import sklearn.ensemble._gb
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.neighbors


# Ensure imports and relative file paths work when launching from api/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from real_estate.utils.forecast_intelligence import ForecastIntelligenceService
from real_estate.utils.market_intelligence import MarketIntelligenceService


app = FastAPI(title="NCR Real Estate API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


CR_FOLDER = "real_estate_data/real_estate_data/circle_rates"
PLOT_TRANSFORMED_FILE = "artifact/data_transformation/cleaned_data/plot_transformed.csv"


def _latest_version_dir(artifact_dir: str) -> str | None:
    """
    Scan *artifact_dir* for sub-folders named v1, v2, v3 … and return the
    path to the highest-numbered one that exists.  Returns None if none found.
    """
    if not os.path.isdir(artifact_dir):
        return None
    versions = []
    for name in os.listdir(artifact_dir):
        m = re.fullmatch(r"v(\d+)", name)
        if m and os.path.isdir(os.path.join(artifact_dir, name)):
            versions.append((int(m.group(1)), name))
    if not versions:
        return None
    _, best = max(versions)
    return os.path.join(artifact_dir, best)


def _resolve_model_path(artifact_dir: str, filename: str, *fallbacks: str) -> str:
    """
    Try the latest versioned dir first (e.g. artifact/apt_model_trainer/v3/),
    then the artifact root dir itself, then each explicit fallback path in order.
    Returns the first path that exists, or the first fallback as a last resort.
    """
    candidates = []
    latest = _latest_version_dir(artifact_dir)
    if latest:
        candidates.append(os.path.join(latest, filename))
    candidates.append(os.path.join(artifact_dir, filename))
    candidates.extend(fallbacks)
    for path in candidates:
        if os.path.exists(path):
            return path
    # Return first fallback so callers can surface a meaningful "not found" error
    return candidates[0]


# ── model paths – always resolve to the latest trained version ──────────────
MODEL_APT_PATH = _resolve_model_path(
    "artifact/apt_model_trainer",
    "best_apt_random_forest.pkl",
    "notebooks/notebooks/sell/apt/best_apt_random_forest.pkl",
)
MODEL_APT_RENT_PATH = _resolve_model_path(
    "artifact/apt_rent_model_trainer",
    "best_apt_random_forest.pkl",
    "notebooks/notebooks/rent/apt/best_apt_random_forest.pkl",
)
MODEL_BF_PATH = _resolve_model_path(
    "artifact/bf_model_trainer",
    "best_bf_random_forest.pkl",
    "notebooks/notebooks/sell/bf/best_bf_random_forest.pkl",
)
MODEL_BF_RENT_PATHS = [
    _resolve_model_path(
        "artifact/bf_rent_model_trainer",
        "best_bf_random_forest.pkl",
        "notebooks/notebooks/rent/bf/best_apt_random_forest.pkl",
    ),
    "notebooks/notebooks/rent/bf/best_apt_random_forest.pkl",
]
MODEL_PLOT_PATHS = [
    _resolve_model_path(
        "artifact/plot_model_trainer",
        "plot_v3_production_model.pkl",
        "notebooks/notebooks/sell/plot/best_plot_random_forest.pkl",
    ),
    "notebooks/notebooks/sell/plot/best_plot_random_forest.pkl",
]
PLOT_FEATURE_COLUMNS_PATH = _resolve_model_path(
    "artifact/plot_model_trainer",
    "plot_feature_columns.pkl",
)
APT_VORONOI_PATH = _resolve_model_path(
    "artifact/apt_model_trainer",
    "apt_vor_kmeans.pkl",
    "notebooks/notebooks/sell/apt/apt_vor_kmeans.pkl",
)
BF_VORONOI_PATH = _resolve_model_path(
    "artifact/bf_model_trainer",
    "bf_vor_kmeans.pkl",
    "notebooks/notebooks/sell/bf/bf_vor_kmeans.pkl",
)
def _discover_road_geojsons() -> list[str]:
    """Auto-discover all road GeoJSON files from known data directories.
    Any *.geojson file whose name contains 'road' (case-insensitive) is included.
    Adding a new city's roads only requires dropping the geojson file in the
    real_estate_data/real_estate_data/ folder — no code changes needed.
    The root directory is only searched as a last-resort fallback so that
    duplicate files at multiple levels are not loaded twice.
    """
    canonical_dirs = [
        "real_estate_data/real_estate_data",
        "real_estate_data",
    ]
    seen: set[str] = set()
    result: list[str] = []

    def _collect(directory: str) -> None:
        if not os.path.isdir(directory):
            return
        for fname in sorted(os.listdir(directory)):
            if not fname.lower().endswith(".geojson"):
                continue
            if "road" not in fname.lower():
                continue
            abs_path = os.path.abspath(os.path.join(directory, fname))
            if abs_path not in seen:
                seen.add(abs_path)
                result.append(abs_path)

    for d in canonical_dirs:
        _collect(d)

    # Only check root directory if nothing found in canonical locations
    if not result:
        _collect(".")

    return result


def _city_display_name(key: str) -> str:
    """Convert an internal city key (snake_case or space-separated) to a UI display name."""
    _OVERRIDES = {
        "new_delhi": "Delhi",
        "new delhi": "Delhi",
        "delhi": "Delhi",
    }
    k = key.strip().lower()
    return _OVERRIDES.get(k, k.replace("_", " ").title())


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _normalize_city_key(city: str) -> str:
    """Normalise a city name to a canonical lowercase key.
    Known aliases are collapsed; any unknown city falls through to its
    normalised form so new cities work without code changes.
    """
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
        "jaipur": "jaipur",
        "pune": "pune",
        "dehradun": "dehradun",
        "aligarh": "aligarh",
        "alwar": "alwar",
        "uttarakhand": "uttarakhand",
    }
    # Explicit map first; unknown cities fall through to the normalised string
    return city_map.get(c, c)


# Token patterns to strip from a circle-rate filename to isolate the city name.
_CR_SUFFIX_RE = re.compile(
    r"(circle[_\s-]*rate[s]?|circle[s]?|_cr\b|\bcr\b|localities|merged|normalized"
    r"|colony[_\s]*list|mcd|_rate[s]?|\(\d+\)|\s+\d+)",
    re.IGNORECASE,
)


def _city_from_filename(filename: str) -> str | None:
    """Infer a city key from a circle-rate JSON filename.
    Hard-coded aliases handle legacy filenames; a generic suffix-stripping
    fallback handles future files automatically.
    """
    f = _norm_text(os.path.splitext(filename)[0])  # drop .json extension first

    # Hard-coded special cases for non-obvious filenames
    if "merged_delhi" in f or f.startswith("delhi"):
        return "delhi"
    if "greater-noida" in f or "greater_noida" in f or "greater noida" in f:
        return "greater noida"
    if "mcd_colony" in f or "missing_circle" in f:
        return None  # not a city-specific file

    # Generic: strip common non-city tokens, normalise separators, look up
    cleaned = _CR_SUFFIX_RE.sub(" ", f)
    cleaned = re.sub(r"[-_]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    return _normalize_city_key(cleaned)


def _build_city_options() -> list[str]:
    """Build the city list dynamically from:
    1. Keys in ncr_colonies.json (locality registry)
    2. Cities inferred from circle-rate JSON filenames
    New cities appear automatically once their data files are added.
    """
    cities_seen: dict[str, str] = {}  # internal_key -> display_name

    # --- Source 1: ncr_colonies.json ---
    colonies_paths = [
        "real_estate_data/real_estate_data/ncr_colonies.json",
        "real_estate_data/ncr_colonies.json",
        "ncr_colonies.json",
    ]
    for cp in colonies_paths:
        if os.path.exists(cp):
            try:
                with open(cp, encoding="utf-8") as f:
                    data = json.load(f)
                for raw_key in data.keys():
                    # Replace underscores before normalising so new_delhi → delhi, etc.
                    key = _normalize_city_key(raw_key.replace("_", " "))
                    if key and key not in cities_seen:
                        cities_seen[key] = _city_display_name(raw_key)
            except Exception:
                pass
            break

    # --- Source 2: circle-rate filenames ---
    cr_dir = CR_FOLDER
    if os.path.isdir(cr_dir):
        for fname in os.listdir(cr_dir):
            if not fname.lower().endswith(".json"):
                continue
            city_key = _city_from_filename(fname.lower())
            if city_key and city_key not in cities_seen:
                cities_seen[city_key] = _city_display_name(city_key)

    if not cities_seen:
        return ["Delhi", "Noida", "Gurgaon", "Faridabad", "Ghaziabad", "Greater Noida", "Jaipur"]

    return sorted(cities_seen.values())


ROAD_GEOJSON_PATHS: list[str] = _discover_road_geojsons()
CITY_OPTIONS: list[str] = _build_city_options()
AGE_CATEGORIES = ["10 to 20 years", "5 to 10 years", "Above 20 years", "Less than 5 years", "New Construction"]
FURNISHING_CATEGORIES = ["Furnished", "Semi-Furnished", "Unfurnished"]
FACING_CATEGORIES = ["East", "North", "North-East", "North-West", "South", "South-East", "South-West", "West"]
FLOOR_LEVELS = ["Low (Ground - 1st)", "Medium (2nd - 7th)", "High (8th+)"]
APT_PROPERTY_SEGMENTS = ["Base", "Mid", "High", "Luxury"]
PLOT_USAGE_OPTIONS = ["Residential", "Commercial"]
PLOT_FACING_OPTIONS = [
    "North",
    "South",
    "East",
    "West",
    "North East",
    "North West",
    "South East",
    "South West",
    "Central",
]
PLOT_ROAD_WIDTH_OPTIONS = ["Upto 9m", "9m to 18m", "18m+"]


_SECTOR_TOKEN_RE = re.compile(r"\bsec(?:tor)?[.\-\s]*(\d{1,3}[a-z]?)\b", re.IGNORECASE)
_LOCALITY_STOPWORDS = {"sector", "sec", "block", "phase", "extension", "extn", "ext"}


def _normalize_property_type(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    key = _norm_text(raw)
    if "res" in key:
        return "Residential"
    if "com" in key:
        return "Commercial"
    if "inst" in key:
        return "Institutional"
    if "agri" in key:
        return "Agricultural"
    if "other" in key:
        return "Other"
    return raw.title()


def _city_rate_candidates(city: str) -> list[str]:
    city_key = _normalize_city_key(city)
    if city_key == "greater noida":
        return ["greater noida", "noida"]
    return [city_key]


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
    q = _norm_text(query)
    c = _norm_text(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 1000.0

    score = max(fuzz.token_set_ratio(q, c), fuzz.token_sort_ratio(q, c), fuzz.partial_ratio(q, c))

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


def load_all_circle_rates() -> dict[str, dict[str, float | dict[str, float]]]:
    by_city: dict[str, dict[str, float | dict[str, float]]] = {}

    def _put(city_key: str, locality: str, rate: float, property_type: str | None = None) -> None:
        c = _normalize_city_key(city_key)
        loc = _norm_text(locality)
        if not c or not loc:
            return

        bucket = by_city.setdefault(c, {})
        prop = _normalize_property_type(property_type)

        if prop:
            existing = bucket.get(loc)
            if isinstance(existing, dict):
                rate_map = existing
            elif isinstance(existing, (int, float)):
                rate_map = {"_default": float(existing)}
            else:
                rate_map = {}

            rate_map[prop] = float(rate)
            if prop == "Residential" or "_default" not in rate_map:
                rate_map["_default"] = float(rate)
            bucket[loc] = rate_map
            return

        existing = bucket.get(loc)
        if isinstance(existing, dict):
            existing.setdefault("_default", float(rate))
        elif existing is None:
            bucket[loc] = float(rate)

    for fpath in glob.glob(os.path.join(CR_FOLDER, "*.json")):
        fname = os.path.basename(fpath).lower()
        file_city = _city_from_filename(fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Canonical format: each record may carry a "city" field.
                    # Fall back to the city inferred from the filename.
                    item_city = file_city or _normalize_city_key(str(item.get("city", "")).strip())
                    if not item_city:
                        continue
                    loc = str(item.get("locality", "")).strip()
                    prop_type = item.get("property_type") or item.get("property_sub_type")
                    rate = item.get("circle_land_cost_inr_per_sqft")
                    if rate is None:
                        rate = item.get("rate_2025_per_sqft")
                    if rate is None:
                        rate = item.get("circle_rate_sqft")
                    if rate is None:
                        rate = item.get("rate")

                    if loc and rate is not None:
                        try:
                            _put(item_city, loc, float(rate), str(prop_type) if prop_type is not None else None)
                        except (TypeError, ValueError):
                            continue

        elif isinstance(data, dict):
            for key, val in data.items():
                key_s = str(key).strip()

                if isinstance(val, (int, float)):
                    if file_city is not None:
                        _put(file_city, key_s, float(val))
                elif isinstance(val, list) and val and isinstance(val[0], dict):
                    if file_city is None:
                        continue
                    for entry in val:
                        chosen = entry.get("circle_rate_sqft")
                        if chosen is None:
                            continue
                        try:
                            _put(file_city, key_s, float(chosen), str(entry.get("property_sub_type", "")))
                        except (TypeError, ValueError):
                            continue
                elif isinstance(val, dict):
                    for sub_loc, sub_val in val.items():
                        if sub_val is not None:
                            try:
                                rate_val = float(sub_val)
                            except (TypeError, ValueError):
                                continue
                            _put(key_s, str(sub_loc), rate_val)

    return by_city


def _get_city_circle_rates(city: str, circle_rates_by_city: dict) -> dict[str, float | dict[str, float]]:
    merged: dict[str, float | dict[str, float]] = {}
    for c in _city_rate_candidates(city):
        bucket = circle_rates_by_city.get(c, {})
        for loc, rate in bucket.items():
            if loc not in merged:
                if isinstance(rate, dict):
                    merged[loc] = {k: float(v) for k, v in rate.items()}
                else:
                    merged[loc] = float(rate)
                continue

            current = merged[loc]
            if isinstance(current, dict) and isinstance(rate, dict):
                for k, v in rate.items():
                    current.setdefault(k, float(v))
            elif isinstance(current, dict) and isinstance(rate, (int, float)):
                current.setdefault("_default", float(rate))
            elif isinstance(current, (int, float)) and isinstance(rate, dict):
                upgraded = {k: float(v) for k, v in rate.items()}
                upgraded.setdefault("_default", float(current))
                merged[loc] = upgraded
    return merged


def _resolve_circle_rate_entry(entry: Any, property_type: str | None = None) -> float | None:
    if isinstance(entry, (int, float)):
        return float(entry)

    if not isinstance(entry, dict) or not entry:
        return None

    if property_type:
        p = _normalize_property_type(property_type)
        if p:
            p_key = _norm_text(p)
            for k, v in entry.items():
                if _norm_text(k) == p_key:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass

    for fallback_key in ("Residential", "_default", "Commercial"):
        if fallback_key in entry:
            try:
                return float(entry[fallback_key])
            except (TypeError, ValueError):
                continue

    for v in entry.values():
        try:
            return float(v)
        except (TypeError, ValueError):
            continue

    return None


def lookup_circle_rate(locality: str, city: str, circle_rates_by_city: dict, property_type: str | None = None) -> float | None:
    if not locality:
        return None

    circle_rates = _get_city_circle_rates(city, circle_rates_by_city)
    if not circle_rates:
        return None

    key = _norm_text(locality)
    if key in circle_rates:
        resolved = _resolve_circle_rate_entry(circle_rates[key], property_type=property_type)
        if resolved is not None:
            return resolved

    best_key, best_score = _best_locality_match(key, list(circle_rates.keys()))
    if best_key is not None and best_score >= 72.0:
        resolved = _resolve_circle_rate_entry(circle_rates[best_key], property_type=property_type)
        if resolved is not None:
            return resolved

    return None


def fuzzy_locality_suggestions(query: str, city: str, circle_rates_by_city: dict, n: int = 8) -> list[str]:
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


def geocode_locality(locality: str, city: str = "Delhi") -> tuple[float, float] | None:
    try:
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="ncr_real_estate_react_api")
        query = f"{locality}, {city}, India"
        loc = geolocator.geocode(query, timeout=5)
        if loc:
            return (loc.latitude, loc.longitude)
        loc = geolocator.geocode(f"{locality}, India", timeout=5)
        return (loc.latitude, loc.longitude) if loc else None
    except Exception:
        return None


def load_plot_locality_encoding() -> dict:
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


def _unwrap_model_bundle(obj):
    """
    Unwrap a bundle dict saved by the new pipeline trainers.
    Returns the raw sklearn model (or pipeline) so callers can call .predict().
    New-style bundles look like: {"model": <estimator>, "kmeans": ..., "features": [...], ...}
    Old-style (notebook) pkl files are already a plain estimator.
    """
    if isinstance(obj, dict):
        model = obj.get("model")
        if model is None:
            for key in ("estimator", "regressor", "pipeline", "final_model", "best_model"):
                candidate = obj.get(key)
                if hasattr(candidate, "predict"):
                    model = candidate
                    break
        return model
    return obj  # already a plain estimator


def load_bf_model():
    if not os.path.exists(MODEL_BF_PATH):
        return None
    return _unwrap_model_bundle(joblib.load(MODEL_BF_PATH))


def load_bf_rent_model():
    for path in MODEL_BF_RENT_PATHS:
        if os.path.exists(path):
            return _unwrap_model_bundle(joblib.load(path))
    return None


def load_apt_model():
    if not os.path.exists(MODEL_APT_PATH):
        return None
    return _unwrap_model_bundle(joblib.load(MODEL_APT_PATH))


def load_apt_rent_model():
    if not os.path.exists(MODEL_APT_RENT_PATH):
        return None
    return _unwrap_model_bundle(joblib.load(MODEL_APT_RENT_PATH))


def load_plot_model_bundle() -> dict:
    model_obj = None
    model_path = None
    for path in MODEL_PLOT_PATHS:
        if os.path.exists(path):
            model_obj = joblib.load(path)
            model_path = path
            break

    if model_obj is None:
        return {"model": None, "features": [], "coord_scaler": None, "kmeans": None, "model_path": None}

    model = None
    features = []
    coord_scaler = None
    kmeans = None

    if isinstance(model_obj, dict):
        model = model_obj.get("model")
        if model is None:
            for key in ("estimator", "regressor", "pipeline", "final_model", "best_model"):
                candidate = model_obj.get(key)
                if hasattr(candidate, "predict"):
                    model = candidate
                    break
        features = list(model_obj.get("features", []) or [])
        coord_scaler = model_obj.get("coord_scaler")
        kmeans = model_obj.get("kmeans")
    else:
        model = model_obj

    if not features and os.path.exists(PLOT_FEATURE_COLUMNS_PATH):
        try:
            loaded = joblib.load(PLOT_FEATURE_COLUMNS_PATH)
            if isinstance(loaded, (list, tuple)):
                features = list(loaded)
        except Exception:
            pass

    if not features and hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)

    return {
        "model": model,
        "features": features,
        "coord_scaler": coord_scaler,
        "kmeans": kmeans,
        "model_path": model_path,
    }


def load_bf_voronoi():
    return joblib.load(BF_VORONOI_PATH) if os.path.exists(BF_VORONOI_PATH) else None


def load_apt_voronoi():
    return joblib.load(APT_VORONOI_PATH) if os.path.exists(APT_VORONOI_PATH) else None


def compute_voronoi_features(lat: float, lon: float, vor_kmeans) -> dict:
    n_cells = vor_kmeans.n_clusters
    point = np.array([[lat, lon]])
    cell_id = int(vor_kmeans.predict(point)[0])
    centers = vor_kmeans.cluster_centers_
    # KMeans was trained on raw (lat, lon) degrees → Euclidean distance is in
    # degrees.  Convert to km so that downstream consumers (uniqueness, road
    # detail) receive a meaningful distance.  1° ≈ 111.32 km at India's lat.
    dist_deg = float(np.sqrt(((point - centers[cell_id]) ** 2).sum()))
    dist_km = dist_deg * 111.32

    feats = {"voronoi_dist_to_seed": dist_km}
    for i in range(n_cells):
        feats[f"vor_cell_{i}"] = 1 if i == cell_id else 0
    return feats


def _road_bucket(properties: dict) -> str | None:
    ref = str(properties.get("ref") or "").upper().replace(" ", "")
    name = str(properties.get("name") or "").upper()
    fclass = str(properties.get("fclass") or "").lower().strip()

    if ref.startswith("MDR") or "MAJOR DISTRICT ROAD" in name:
        return "MDR"
    if ref.startswith("SH") or re.search(r"\bSH\s*\d", name):
        return "SH"
    if ref.startswith("NH") or re.search(r"\bNH\s*\d", name):
        return "NH"
    # fclass-based fallback for roads without a ref code (Pune / Jaipur etc.)
    if fclass in ("motorway", "trunk"):
        return "NH"
    if fclass in ("primary", "primary_link"):
        return "SH"
    if fclass in ("secondary", "secondary_link"):
        return "MDR"
    return None


def load_road_segments() -> dict[str, np.ndarray]:
    # Deduplicate paths so the same file isn't loaded twice
    seen: set[str] = set()
    available_paths: list[str] = []
    for path in ROAD_GEOJSON_PATHS:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and abs_path not in seen:
            seen.add(abs_path)
            available_paths.append(abs_path)

    if not available_paths:
        return {
            "MDR": np.empty((0, 4), dtype=np.float64),
            "SH": np.empty((0, 4), dtype=np.float64),
            "NH": np.empty((0, 4), dtype=np.float64),
        }

    segments = {"MDR": [], "SH": [], "NH": []}
    for geojson_path in available_paths:
        with open(geojson_path, "r", encoding="utf-8") as f:
            gj = json.load(f)

        for feature in gj.get("features", []):
            props = feature.get("properties", {})
            geometry = feature.get("geometry", {})
            bucket = _road_bucket(props)
            if bucket is None:
                continue

            gtype = geometry.get("type")
            coords = geometry.get("coordinates", [])
            if gtype == "LineString":
                lines = [coords]
            elif gtype == "MultiLineString":
                lines = coords
            else:
                continue

            for line in lines:
                if not line or len(line) < 2:
                    continue
                for (lon1, lat1), (lon2, lat2) in zip(line[:-1], line[1:]):
                    segments[bucket].append((lat1, lon1, lat2, lon2))

    out = {}
    for key, vals in segments.items():
        out[key] = np.array(vals, dtype=np.float64) if vals else np.empty((0, 4), dtype=np.float64)
    return out


def nearest_distance_km(lat: float, lon: float, seg_arr: np.ndarray) -> float:
    if seg_arr.size == 0:
        return float("nan")

    lat1 = seg_arr[:, 0]
    lon1 = seg_arr[:, 1]
    lat2 = seg_arr[:, 2]
    lon2 = seg_arr[:, 3]

    k_lat = 110.574
    k_lon = 111.320 * np.cos(np.radians(lat))

    x1 = (lon1 - lon) * k_lon
    y1 = (lat1 - lat) * k_lat
    x2 = (lon2 - lon) * k_lon
    y2 = (lat2 - lat) * k_lat

    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy

    t = np.zeros_like(denom)
    valid = denom > 0
    t[valid] = -(x1[valid] * dx[valid] + y1[valid] * dy[valid]) / denom[valid]
    t = np.clip(t, 0.0, 1.0)

    px = x1 + t * dx
    py = y1 + t * dy
    d = np.sqrt(px * px + py * py)
    return float(np.min(d))


def get_spatial_cluster_features(lat: float, lon: float, model_bundle: dict, feature_columns: list[str]) -> tuple[dict[str, float], float, int | None]:
    cluster_cols = sorted(
        [c for c in feature_columns if re.fullmatch(r"c_\d+", c)],
        key=lambda x: int(x.split("_")[1]),
    )
    cluster_features = {col: 0.0 for col in cluster_cols}
    dist_to_center = 0.0
    cluster_id = None

    kmeans_model = model_bundle.get("kmeans")
    coord_scaler = model_bundle.get("coord_scaler")

    if not cluster_cols or kmeans_model is None:
        return cluster_features, dist_to_center, cluster_id

    coords_df = pd.DataFrame({"latitude": [float(lat)], "longitude": [float(lon)]})

    try:
        coords_for_cluster = coord_scaler.transform(coords_df) if coord_scaler is not None else coords_df.to_numpy(dtype=float)
        cluster_id = int(kmeans_model.predict(coords_for_cluster)[0])
        center = kmeans_model.cluster_centers_[cluster_id]
        point = np.asarray(coords_for_cluster, dtype=float)[0]
        dist_to_center = float(np.linalg.norm(point - center))

        cluster_col = f"c_{cluster_id}"
        if cluster_col in cluster_features:
            cluster_features[cluster_col] = 1.0
    except Exception:
        pass

    return cluster_features, dist_to_center, cluster_id


def _ohe(value: str, categories: list[str], prefix: str) -> dict[str, int]:
    return {f"{prefix}_{c}": 1 if c == value else 0 for c in categories}


def _safe_num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(out) or np.isinf(out):
        return default
    return out


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _forecast_qoq_volatility_pct(quarter_rows: list[dict[str, Any]]) -> float:
    points = [
        _safe_num(row.get("forecast_price_per_sqft"), default=np.nan)
        for row in quarter_rows
    ]
    points = [v for v in points if np.isfinite(v) and v > 0]
    if len(points) < 3:
        return 0.0

    arr = np.array(points, dtype=float)
    qoq_pct = (arr[1:] - arr[:-1]) / np.clip(arr[:-1], 1e-9, None) * 100.0
    if len(qoq_pct) == 0:
        return 0.0
    return float(np.std(qoq_pct, ddof=0))


def _extract_horizon_ppsf(quarter_rows: list[dict[str, Any]], hold_years: int) -> tuple[float, int]:
    if not quarter_rows:
        return 0.0, 0

    horizon_q = max(1, hold_years * 4)
    horizon_q = min(horizon_q, len(quarter_rows))
    row = quarter_rows[horizon_q - 1]
    ppsf = _safe_num(row.get("forecast_price_per_sqft"), default=0.0)
    return ppsf, horizon_q


# ─────────────────────────────────────────────────────────────
# XAI – Explainable Prediction Report
# ─────────────────────────────────────────────────────────────

def _compute_ncr_stats() -> dict[str, dict]:
    """Compute NCR-wide reference stats for XAI normalisation (runs once at startup)."""
    segment_files = {
        "builder-floor": "inputs/builder_floor_with_pi.csv",
        "apartment": "inputs/apartment_with_pi.csv",
        "plot": "inputs/plot_with_pi.csv",
    }
    results: dict[str, dict] = {}
    for seg, fpath in segment_files.items():
        try:
            wanted = {"price_per_sqft", "circle_rate", "covered_area_sqft", "plot_area"}
            df = pd.read_csv(fpath, usecols=lambda c: c in wanted)
            area_col = "covered_area_sqft" if "covered_area_sqft" in df.columns else "plot_area"
            results[seg] = {
                "median_ppsf": float(pd.to_numeric(df.get("price_per_sqft"), errors="coerce").dropna().median()) if "price_per_sqft" in df else 10000.0,
                "median_cr": float(pd.to_numeric(df.get("circle_rate"), errors="coerce").dropna().median()) if "circle_rate" in df else 8000.0,
                "median_area": float(pd.to_numeric(df.get(area_col), errors="coerce").dropna().median()) if area_col in df.columns else 1200.0,
            }
        except Exception:
            results[seg] = {"median_ppsf": 10000.0, "median_cr": 8000.0, "median_area": 1200.0}
    return results


def _xai_score(value: float, breakpoints: list[tuple[float, int]]) -> int:
    """Assign an integer score given a value and descending (threshold, score) breakpoints."""
    for threshold, score in breakpoints:
        if value >= threshold:
            return score
    return breakpoints[-1][1]


def _build_xai_explanation(
    *,
    segment: str,
    ppsf: float,
    circle_rate: float,
    area_sqft: float,
    # BF / APT
    pred_ratio: float | None = None,
    voronoi_dist: float | None = None,
    is_main_road: int = 0,
    furnishing: str = "Semi-Furnished",
    age: str = "5 to 10 years",
    floor_level: str | None = None,
    property_segment: str | None = None,
    is_parking: int = 0,
    is_pool: int = 0,
    is_garden_park: int = 0,
    is_gated: int = 0,
    is_corner: int = 0,
    # Plot
    nh_km: float | None = None,
    sh_km: float | None = None,
    mdr_km: float | None = None,
    road_width_18_plus: int = 0,
    road_width_9_to_18m: int = 0,
    usage_type: str = "Residential",
    has_boundary_wall: int = 0,
    is_park_facing: int = 0,
    is_rectangular: int = 1,
    # Reference + growth
    ncr_stats: dict | None = None,
    locality_yoy_pct: float | None = None,
) -> dict[str, Any]:
    seg = segment.strip().lower().replace("_", "-")
    stats = (ncr_stats or {}).get(seg, {})
    ncr_cr = float(stats.get("median_cr", 8000.0))
    ncr_ppsf = float(stats.get("median_ppsf", 10000.0))
    ncr_area = float(stats.get("median_area", 1200.0))

    drivers: list[dict[str, Any]] = []

    # ── 1. Location / Market Zone ──────────────────────────────
    cr_ratio = circle_rate / ncr_cr if ncr_cr > 0 else 1.0
    loc_score = _xai_score(cr_ratio, [(2.0, 2), (1.3, 1), (0.7, 0), (0.4, -1), (0.0, -2)])
    loc_msgs = {
        2: f"Circle rate INR {circle_rate:,.0f}/sqft is {cr_ratio:.1f}× segment median — premium market zone.",
        1: f"Circle rate INR {circle_rate:,.0f}/sqft is above segment median (INR {ncr_cr:,.0f}/sqft) — above-average zone.",
        0: f"Circle rate INR {circle_rate:,.0f}/sqft is in line with segment median — typical NCR locality.",
        -1: f"Circle rate INR {circle_rate:,.0f}/sqft is below segment median (INR {ncr_cr:,.0f}/sqft) — budget or emerging zone.",
        -2: f"Circle rate INR {circle_rate:,.0f}/sqft is well below segment median — peripheral or low-demand zone.",
    }
    drivers.append({"key": "location", "label": "Location / Market Zone", "score": loc_score, "detail": loc_msgs[loc_score]})

    # ── 2. Circle-Rate Alignment (market premium over government floor) ──
    cr_val_ratio = pred_ratio if pred_ratio is not None else (ppsf / circle_rate if circle_rate > 0 else 1.0)
    cra_score = _xai_score(cr_val_ratio, [(1.5, 2), (1.2, 1), (0.85, 0), (0.65, -1), (0.0, -2)])
    cra_msgs = {
        2: f"Property trades at {cr_val_ratio:.2f}× circle rate — strong market demand above government floor value.",
        1: f"Priced at {cr_val_ratio:.2f}× circle rate — healthy demand, moderate premium over floor value.",
        0: f"Aligned with circle rate ({cr_val_ratio:.2f}×) — fairly valued relative to government floor.",
        -1: f"Below circle rate ({cr_val_ratio:.2f}×) — slight discount, may reflect condition or soft demand.",
        -2: f"Significantly below circle rate ({cr_val_ratio:.2f}×) — deep discount, warrants scrutiny.",
    }
    drivers.append({"key": "circle_rate", "label": "Circle-Rate Impact", "score": cra_score, "detail": cra_msgs[cra_score]})

    # ── 3. Area Impact ──────────────────────────────────────────
    area_ratio = area_sqft / ncr_area if ncr_area > 0 else 1.0
    area_score = _xai_score(area_ratio, [(2.0, 2), (1.4, 1), (0.7, 0), (0.4, -1), (0.0, -2)])
    area_label = "Plot Area" if seg == "plot" else "Built-Up Area"
    area_msgs = {
        2: f"{area_sqft:,.0f} sqft is {area_ratio:.1f}× median — significantly large, commanding size premium.",
        1: f"{area_sqft:,.0f} sqft is above median ({ncr_area:,.0f} sqft) — above-average size with premium.",
        0: f"{area_sqft:,.0f} sqft is near median ({ncr_area:,.0f} sqft) — typical size, neutral impact.",
        -1: f"{area_sqft:,.0f} sqft is below median ({ncr_area:,.0f} sqft) — compact property, size discount.",
        -2: f"{area_sqft:,.0f} sqft is well below median ({ncr_area:,.0f} sqft) — very small, significant discount.",
    }
    drivers.append({"key": "area", "label": area_label, "score": area_score, "detail": area_msgs[area_score]})

    # ── 4. Road Connectivity ────────────────────────────────────
    if seg == "plot" and nh_km is not None and sh_km is not None and mdr_km is not None:
        min_km = min(nh_km, sh_km, mdr_km)
        road_score = _xai_score(min_km, [(0.0, 2)])  # will be overridden below
        if min_km < 1.5: road_score = 2
        elif min_km < 5.0: road_score = 1
        elif min_km < 12.0: road_score = 0
        elif min_km < 20.0: road_score = -1
        else: road_score = -2
        road_detail = f"Nearest highway/arterial is {min_km:.1f} km (NH {nh_km:.1f} km · SH {sh_km:.1f} km · MDR {mdr_km:.1f} km)."
        if road_width_18_plus:
            road_score = min(2, road_score + 1)
            road_detail += " Wide approach road (18m+) adds further connectivity premium."
        elif road_width_9_to_18m:
            road_detail += " Moderate approach road width (9–18m)."
        else:
            road_score = max(-2, road_score - 1)
            road_detail += " Narrow approach road (<9m) limits accessibility."
    else:
        if is_main_road:
            road_score, road_detail = 2, "On or directly accessible from a main road — excellent connectivity."
        elif voronoi_dist is not None and voronoi_dist < 2.0:
            road_score = 1
            road_detail = f"Close to a dense urban cluster ({voronoi_dist:.1f} km) — good road network access."
        elif voronoi_dist is not None and voronoi_dist > 6.0:
            road_score = -1
            road_detail = f"Located {voronoi_dist:.1f} km from nearest dense cluster — connectivity may be limited."
        else:
            road_score, road_detail = 0, "Moderate connectivity — no main road frontage, typical suburban access."

    drivers.append({"key": "connectivity", "label": "Road Connectivity", "score": road_score, "detail": road_detail})

    # ── 5. Property Quality / Type ──────────────────────────────
    q = 0
    q_tags: list[str] = []

    if seg == "plot":
        if usage_type and "comm" in usage_type.lower():
            q += 1; q_tags.append("commercial usage")
        if road_width_18_plus:
            q += 1; q_tags.append("wide road frontage")
        if is_gated:
            q += 1; q_tags.append("gated")
        if is_park_facing:
            q += 1; q_tags.append("park-facing")
        if not is_rectangular:
            q -= 1; q_tags.append("irregular shape")
    else:
        seg_map = {"luxury": 2, "high": 1, "mid": 0, "base": -1}
        q += seg_map.get((property_segment or "mid").lower(), 0)
        if property_segment and property_segment.lower() in ("luxury", "high", "base"):
            q_tags.append(f"{property_segment.lower()} segment")
        if floor_level:
            fl = floor_level.lower()
            if "high" in fl or "8th" in fl:
                q += 1; q_tags.append("high floor")
            elif "low" in fl or "ground" in fl:
                q -= 1; q_tags.append("low/ground floor")
        furn_map = {"furnished": 1, "unfurnished": -1}
        q += furn_map.get(furnishing.lower(), 0)
        if furnishing.lower() in furn_map:
            q_tags.append(furnishing.lower())
        if "new construction" in age.lower() or "less than 5" in age.lower():
            q += 1; q_tags.append("new construction")
        elif "above 20" in age.lower():
            q -= 1; q_tags.append("aged >20 years")
        amenities = int(is_parking) + int(is_pool) + int(is_garden_park) + int(is_gated) + int(is_corner)
        if amenities >= 4:
            q += 1; q_tags.append("strong amenities")
        elif amenities <= 1:
            q -= 1; q_tags.append("minimal amenities")

    q = max(-2, min(2, q))
    tag_str = (", ".join(q_tags[:3]) + ".") if q_tags else "standard attributes."
    quality_detail = {
        2: f"Premium quality — {tag_str}",
        1: f"Above-average quality — {tag_str}",
        0: f"Standard quality — {tag_str}",
        -1: f"Below-average quality — {tag_str}",
        -2: f"Low quality rating — {tag_str}",
    }[q]
    drivers.append({"key": "property_type", "label": "Property-Type Quality", "score": q, "detail": quality_detail})

    # ── 6. Market Growth ────────────────────────────────────────
    if locality_yoy_pct is not None:
        g = _xai_score(locality_yoy_pct, [(8.0, 2), (5.0, 1), (2.0, 0), (0.0, -1), (-99.0, -2)])
        g_msgs = {
            2: f"Locality forecast YoY is {locality_yoy_pct:.1f}% — high-growth zone, strong appreciation signal.",
            1: f"Forecast YoY growth {locality_yoy_pct:.1f}% — healthy appreciation trajectory.",
            0: f"Moderate forecast growth {locality_yoy_pct:.1f}% YoY — stable market.",
            -1: f"Subdued forecast growth {locality_yoy_pct:.1f}% YoY — watch for price stagnation.",
            -2: f"Negative forecast trajectory {locality_yoy_pct:.1f}% YoY — potential price decline risk.",
        }
        growth_detail, unavailable = g_msgs[g], False
    else:
        g, growth_detail, unavailable = 0, "Locality growth data not available — use Forecast tab for YoY trend.", True

    drivers.append({"key": "market_growth", "label": "Market Growth", "score": g, "detail": growth_detail, "unavailable": unavailable})

    # ── 7. Listing vs Segment Median ────────────────────────────
    ppsf_ratio = ppsf / ncr_ppsf if ncr_ppsf > 0 else 1.0
    gap_score = _xai_score(ppsf_ratio, [(1.8, 2), (1.25, 1), (0.75, 0), (0.5, -1), (0.0, -2)])
    gap_msgs = {
        2: f"Predicted INR {ppsf:,.0f}/sqft is {ppsf_ratio:.1f}× segment median — strongly premium valuation.",
        1: f"Predicted INR {ppsf:,.0f}/sqft is above segment median (INR {ncr_ppsf:,.0f}/sqft) — above-market pricing.",
        0: f"Predicted INR {ppsf:,.0f}/sqft is near segment median (INR {ncr_ppsf:,.0f}/sqft) — fair market pricing.",
        -1: f"Predicted INR {ppsf:,.0f}/sqft is below segment median (INR {ncr_ppsf:,.0f}/sqft) — below-market pricing.",
        -2: f"Predicted INR {ppsf:,.0f}/sqft is well below segment median — deeply discounted or peripheral.",
    }
    drivers.append({"key": "listing_gap", "label": "Listing vs Segment Median", "score": gap_score, "detail": gap_msgs[gap_score]})

    # ── Narrative ───────────────────────────────────────────────
    top3 = sorted(drivers, key=lambda d: abs(d.get("score", 0)), reverse=True)[:3]
    pos_top = [d for d in top3 if d["score"] > 0]
    neg_top = [d for d in top3 if d["score"] < 0]
    seg_label = {"builder-floor": "builder floor property", "apartment": "apartment", "plot": "plot"}.get(seg, "property")
    vs_median = "above" if ppsf > ncr_ppsf else ("below" if ppsf < ncr_ppsf * 0.95 else "aligned with")

    if pos_top and neg_top:
        pos_desc = " and ".join(d["label"].lower() for d in pos_top[:2])
        neg_desc = neg_top[0]["label"].lower()
        narrative = (
            f'This {seg_label} is valued {vs_median} the segment median primarily due to {pos_desc}, '
            f'while {neg_desc} moderates the valuation.'
        )
    elif pos_top:
        pos_desc = ", ".join(d["label"].lower() for d in pos_top[:3])
        narrative = f"This {seg_label} commands a premium mainly because of strong {pos_desc}."
    elif neg_top:
        neg_desc = ", ".join(d["label"].lower() for d in neg_top[:3])
        narrative = f"This {seg_label} is priced below the segment median mainly due to weaker {neg_desc}."
    else:
        narrative = f"This {seg_label} is fairly valued — most factors are aligned close to segment medians."

    return {
        "drivers": drivers,
        "narrative": narrative,
        "ppsfVsMedianPct": round((ppsf / ncr_ppsf - 1.0) * 100, 1) if ncr_ppsf > 0 else None,
        "ncrReference": {
            "medianPpsf": round(ncr_ppsf, 0),
            "medianCr": round(ncr_cr, 0),
            "medianArea": round(ncr_area, 0),
        },
    }


def _build_buy_decision_payload(context_payload: dict[str, Any], hold_years: int) -> dict[str, Any]:
    kpis = context_payload.get("kpis", {})
    listing_ppsf = _safe_num(kpis.get("listingPricePpsf"), default=0.0)
    model_ppsf = _safe_num(kpis.get("modelPricePpsf"), default=0.0)
    valuation_gap_pct = _safe_num(kpis.get("deltaPct"), default=0.0)

    quarter_rows = context_payload.get("quarterTable", []) or []
    horizon_ppsf, horizon_quarter_used = _extract_horizon_ppsf(quarter_rows, hold_years=hold_years)
    expected_upside_pct = ((horizon_ppsf / listing_ppsf) - 1.0) * 100.0 if listing_ppsf > 0 else 0.0

    yoy_rows = context_payload.get("yoy", {}).get("property", []) or []
    yoy_vals = [_safe_num(row.get("yoy_pct"), default=np.nan) for row in yoy_rows]
    yoy_vals = [v for v in yoy_vals if np.isfinite(v)]
    avg_yoy_pct = float(np.mean(yoy_vals)) if yoy_vals else 0.0

    volatility_pct = _forecast_qoq_volatility_pct(quarter_rows)

    valuation_score = _clamp(50.0 + (-valuation_gap_pct * 2.0), 0.0, 100.0)
    growth_score = _clamp(50.0 + (avg_yoy_pct * 3.0), 0.0, 100.0)
    upside_score = _clamp(50.0 + (expected_upside_pct * 1.5), 0.0, 100.0)
    risk_penalty = _clamp(volatility_pct * 2.0, 0.0, 30.0)

    overall_score = _clamp(
        0.40 * valuation_score + 0.35 * growth_score + 0.25 * upside_score - risk_penalty,
        0.0,
        100.0,
    )

    confidence = 0.55
    confidence += min(len(quarter_rows), 20) / 20.0 * 0.25
    confidence += 0.10 if not bool(context_payload.get("trendIsCityFallback", False)) else -0.10
    confidence = _clamp(confidence, 0.35, 0.95)

    if overall_score >= 68.0:
        recommendation = "Buy"
    elif overall_score >= 50.0:
        recommendation = "Hold / Neutral"
    else:
        recommendation = "Avoid"

    reasons: list[str] = []
    risks: list[str] = []

    if valuation_gap_pct <= -5.0:
        reasons.append(f"Listing is below model fair value by {abs(valuation_gap_pct):.2f}%.")
    elif valuation_gap_pct >= 5.0:
        risks.append(f"Listing is above model fair value by {valuation_gap_pct:.2f}%.")

    if expected_upside_pct >= 10.0:
        reasons.append(f"Projected {hold_years}-year upside is {expected_upside_pct:.2f}% based on forecast trajectory.")
    elif expected_upside_pct <= 0.0:
        risks.append(f"Projected {hold_years}-year upside is weak at {expected_upside_pct:.2f}%.")

    if avg_yoy_pct >= 5.0:
        reasons.append(f"Average YoY trend signal is healthy at {avg_yoy_pct:.2f}%.")
    elif avg_yoy_pct <= 1.0:
        risks.append(f"Average YoY trend is soft at {avg_yoy_pct:.2f}%.")

    if volatility_pct >= 6.0:
        risks.append(f"Forecast quarter-over-quarter volatility is elevated ({volatility_pct:.2f}% stdev).")
    elif volatility_pct <= 2.5:
        reasons.append(f"Forecast quarter-over-quarter volatility is contained ({volatility_pct:.2f}% stdev).")

    if not reasons:
        reasons.append("Balanced valuation and trend signals with no dominant upside trigger.")
    if not risks:
        risks.append("No major risk flags triggered from current valuation and trend metrics.")

    return {
        "recommendation": recommendation,
        "score": round(overall_score, 2),
        "confidence": round(confidence, 2),
        "holdYears": hold_years,
        "valuationGapPct": round(valuation_gap_pct, 2),
        "expectedUpsidePct": round(expected_upside_pct, 2),
        "avgYoYPct": round(avg_yoy_pct, 2),
        "forecastVolatilityPct": round(volatility_pct, 2),
        "horizonQuarterUsed": horizon_quarter_used,
        "reasons": reasons[:3],
        "risks": risks[:3],
    }


def _build_roi_payload(
    context_payload: dict[str, Any],
    *,
    hold_years: int,
    area_sqft_override: float | None,
    purchase_cost_pct: float,
    annual_holding_cost_pct: float,
    exit_cost_pct: float,
    rent_yield_pct: float | None,
) -> dict[str, Any]:
    kpis = context_payload.get("kpis", {})
    prop = context_payload.get("property", {})
    quarter_rows = context_payload.get("quarterTable", []) or []

    buy_ppsf = _safe_num(kpis.get("listingPricePpsf"), default=0.0)
    if buy_ppsf <= 0:
        buy_ppsf = _safe_num(kpis.get("modelPricePpsf"), default=0.0)

    area_sqft = _safe_num(area_sqft_override, default=0.0)
    if area_sqft <= 0:
        area_sqft = _safe_num(prop.get("covered_area_sqft"), default=0.0)
    if area_sqft <= 0:
        area_sqft = 1000.0

    exit_ppsf, horizon_quarter_used = _extract_horizon_ppsf(quarter_rows, hold_years=hold_years)
    if exit_ppsf <= 0:
        exit_ppsf = buy_ppsf

    buy_price = buy_ppsf * area_sqft
    purchase_costs = buy_price * (purchase_cost_pct / 100.0)
    total_invested = buy_price + purchase_costs

    gross_sale_price = exit_ppsf * area_sqft
    exit_costs = gross_sale_price * (exit_cost_pct / 100.0)
    net_sale_proceeds = gross_sale_price - exit_costs

    segment_key = _norm_text(context_payload.get("segment", ""))
    monthly_rent_estimate: float | None = None
    annual_rent_estimate = 0.0

    if rent_yield_pct is not None:
        effective_rent_yield_pct = max(0.0, _safe_num(rent_yield_pct, default=0.0))
        rent_yield_source = "manual_override"
        annual_rent_estimate = buy_price * (effective_rent_yield_pct / 100.0)
        monthly_rent_estimate = annual_rent_estimate / 12.0 if annual_rent_estimate > 0 else None
    elif segment_key in {"builder-floor", "apartment"}:
        prop = context_payload.get("property", {})
        bhk = int(_clamp(round(_safe_num(prop.get("bhk"), default=2.0)), 1, 10))
        bathrooms = int(_clamp(round(_safe_num(prop.get("bathrooms"), default=bhk)), 1, 10))
        balconies = int(_clamp(round(_safe_num(prop.get("balconies"), default=max(1, bhk - 1))), 0, 10))
        circle_rate = _safe_num(prop.get("circle_rate"), default=0.0)
        if circle_rate <= 0:
            circle_rate = buy_ppsf

        common_kwargs = {
            "bhk": bhk,
            "area_sqft": area_sqft,
            "bathrooms": bathrooms,
            "balconies": balconies,
            "circle_rate": circle_rate,
            "is_parking": int(_clamp(_safe_num(prop.get("is_parking"), default=1.0), 0, 1)),
            "is_pool": 0,
            "is_main_road": 0,
            "is_garden_park": 0,
            "is_gated": int(_clamp(_safe_num(prop.get("is_gated"), default=0.0), 0, 1)),
            "is_corner": int(_clamp(_safe_num(prop.get("is_corner"), default=0.0), 0, 1)),
            "age": "5 to 10 years",
            "furnishing": "Semi-Furnished",
            "facing": "North",
            "voronoi_feats": {"voronoi_dist_to_seed": 0.0},
        }

        rent_model = bf_rent_model if segment_key == "builder-floor" else apt_rent_model
        try:
            if rent_model is not None:
                if segment_key == "builder-floor":
                    floor_feats = {
                        "current_floor": 1,
                        "total_floors": max(2, bhk + 1),
                        "is_ground_floor": 0,
                        "is_top_floor": 0,
                        "is_basement": 0,
                    }
                    rent_X = build_features(model=rent_model, floor_feats=floor_feats, **common_kwargs)
                else:
                    floor_feats = {
                        "floor_low": 0,
                        "floor_medium": 1,
                        "floor_high": 0,
                        "is_ground_floor": 0,
                        "is_top_floor": 0,
                    }
                    floor_feats.update(build_apartment_property_features("Mid", rent_model))
                    rent_X = build_features(model=rent_model, floor_feats=floor_feats, **common_kwargs)

                rent_pred_log = float(rent_model.predict(rent_X)[0])
                monthly_rent_estimate = max(0.0, float(np.expm1(rent_pred_log)))
                annual_rent_estimate = monthly_rent_estimate * 12.0
                effective_rent_yield_pct = ((annual_rent_estimate / buy_price) * 100.0) if buy_price > 0 else 0.0
                rent_yield_source = "rent_model"
            else:
                effective_rent_yield_pct = 0.0
                rent_yield_source = "no_rent_assumption"
        except Exception:
            effective_rent_yield_pct = 0.0
            monthly_rent_estimate = None
            annual_rent_estimate = 0.0
            rent_yield_source = "no_rent_assumption"
    else:
        effective_rent_yield_pct = 0.0
        rent_yield_source = "no_rent_assumption"

    rental_income_total = annual_rent_estimate * hold_years
    holding_costs_total = buy_price * (annual_holding_cost_pct / 100.0) * hold_years

    net_profit = net_sale_proceeds + rental_income_total - holding_costs_total - total_invested
    roi_pct = (net_profit / total_invested) * 100.0 if total_invested > 0 else 0.0

    payoff_multiple = (
        (net_sale_proceeds + rental_income_total - holding_costs_total) / total_invested
        if total_invested > 0
        else 1.0
    )
    annualized_cagr_pct = 0.0
    if hold_years > 0 and payoff_multiple > 0:
        annualized_cagr_pct = (pow(payoff_multiple, 1.0 / hold_years) - 1.0) * 100.0

    if roi_pct >= 35.0:
        verdict = "Strong"
    elif roi_pct >= 15.0:
        verdict = "Moderate"
    else:
        verdict = "Weak"

    return {
        "inputs": {
            "holdYears": hold_years,
            "areaSqft": round(area_sqft, 2),
            "purchaseCostPct": purchase_cost_pct,
            "annualHoldingCostPct": annual_holding_cost_pct,
            "exitCostPct": exit_cost_pct,
            "rentYieldPct": round(effective_rent_yield_pct, 3),
            "rentYieldSource": rent_yield_source,
        },
        "valuation": {
            "buyPpsf": round(buy_ppsf, 2),
            "forecastExitPpsf": round(exit_ppsf, 2),
            "horizonQuarterUsed": horizon_quarter_used,
        },
        "cashflows": {
            "buyPrice": round(buy_price, 2),
            "purchaseCosts": round(purchase_costs, 2),
            "totalInvested": round(total_invested, 2),
            "grossSalePrice": round(gross_sale_price, 2),
            "exitCosts": round(exit_costs, 2),
            "netSaleProceeds": round(net_sale_proceeds, 2),
            "rentalIncomeTotal": round(rental_income_total, 2),
            "holdingCostsTotal": round(holding_costs_total, 2),
            "netProfit": round(net_profit, 2),
        },
        "rentalYield": {
            "source": rent_yield_source,
            "effectivePct": round(effective_rent_yield_pct, 3),
            "monthlyRentEstimate": round(monthly_rent_estimate, 2) if monthly_rent_estimate is not None else None,
            "annualRentEstimate": round(annual_rent_estimate, 2),
        },
        "returns": {
            "roiPct": round(roi_pct, 2),
            "annualizedCagrPct": round(annualized_cagr_pct, 2),
            "payoffMultiple": round(payoff_multiple, 3),
        },
        "verdict": verdict,
    }


def build_features(
    *,
    bhk: int,
    area_sqft: float,
    bathrooms: int,
    balconies: int,
    circle_rate: float,
    is_parking: int,
    is_pool: int,
    is_main_road: int,
    is_garden_park: int,
    is_gated: int,
    is_corner: int,
    age: str,
    furnishing: str,
    facing: str,
    voronoi_feats: dict,
    model,
    floor_feats: dict | None = None,
) -> pd.DataFrame:
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
    base.update(_ohe(age, AGE_CATEGORIES, "age"))
    base.update(_ohe(furnishing, FURNISHING_CATEGORIES, "furn"))
    base.update(_ohe(facing, FACING_CATEGORIES, "facing"))
    base.update(voronoi_feats)
    if floor_feats:
        base.update(floor_feats)

    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        row = {c: base.get(c, 0) for c in cols}
        return pd.DataFrame([row], columns=cols)

    return pd.DataFrame([base])


def build_apartment_property_features(property_segment: str, model) -> dict:
    segment = (property_segment or "").strip().lower()
    model_cols = set(getattr(model, "feature_names_in_", [])) if model is not None else set()

    feats = {
        "property_type_base": 0,
        "property_type_mid": 0,
        "property_type_high": 0,
        "property_type_luxury": 0,
        "property_tier_base": 0,
        "property_tier_mid": 0,
        "property_tier_high": 0,
        "property_tier_luxury": 0,
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
        feats["property_type_mid"] = 1
        feats["property_tier_mid"] = 1
        feats["locality_tier_mid"] = 1
        feats["locality_tier_ord"] = 1

    return feats


def build_plot_features(
    *,
    area_sqft: float,
    circle_rate: float,
    latitude: float,
    longitude: float,
    usage_type: str,
    facing_direction: str,
    is_park_facing: int,
    is_corner: int,
    is_rectangular: int,
    is_gated: int,
    has_boundary_wall: int,
    road_width_upto_9m: int,
    road_width_9_to_18m: int,
    road_width_18_plus: int,
    model_bundle: dict,
    road_segments: dict[str, np.ndarray],
) -> pd.DataFrame:
    model = model_bundle.get("model")
    feature_columns = list(model_bundle.get("features") or [])

    if not feature_columns and hasattr(model, "feature_names_in_"):
        feature_columns = list(model.feature_names_in_)

    row = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    closest_distance_mdr_km = nearest_distance_km(float(latitude), float(longitude), road_segments["MDR"])
    closest_distance_sh_km = nearest_distance_km(float(latitude), float(longitude), road_segments["SH"])
    closest_distance_nh_km = nearest_distance_km(float(latitude), float(longitude), road_segments["NH"])

    cluster_features, dist_to_center, _ = get_spatial_cluster_features(float(latitude), float(longitude), model_bundle, feature_columns)

    numeric_values = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "circle_rate": float(circle_rate),
        "closest_distance_MDR_km": 0.0 if np.isnan(closest_distance_mdr_km) else float(closest_distance_mdr_km),
        "closest_distance_SH_km": 0.0 if np.isnan(closest_distance_sh_km) else float(closest_distance_sh_km),
        "closest_distance_NH_km": 0.0 if np.isnan(closest_distance_nh_km) else float(closest_distance_nh_km),
        "log_plot_area": float(np.log1p(area_sqft)),
        "dist_to_center": float(dist_to_center),
        "is_park_facing": float(is_park_facing),
        "is_corner": float(is_corner),
        "is_rectangular": float(is_rectangular),
        "is_gated": float(is_gated),
        "has_boundary_wall": float(has_boundary_wall),
        "road_width_upto_9m": float(road_width_upto_9m),
        "road_width_9_to_18m": float(road_width_9_to_18m),
        "road_width_18_plus": float(road_width_18_plus),
    }

    for col, val in numeric_values.items():
        if col in row.columns:
            row.at[0, col] = val

    usage_col = f"usage_type_{usage_type}"
    facing_col = f"facing_direction_{facing_direction}"
    if usage_col in row.columns:
        row.at[0, usage_col] = 1.0
    if facing_col in row.columns:
        row.at[0, facing_col] = 1.0

    for col, val in cluster_features.items():
        if col in row.columns:
            row.at[0, col] = float(val)

    return row


class SharedPredictRequest(BaseModel):
    bhk: int = Field(3, ge=1, le=10)
    area_sqft: float = Field(1200.0, gt=0)
    bathrooms: int = Field(2, ge=1, le=10)
    balconies: int = Field(1, ge=0, le=10)
    age: str = Field("5 to 10 years")
    furnishing: str = Field("Semi-Furnished")
    facing: str = Field("North")
    circle_rate: float = Field(11891.59, gt=0)
    is_parking: int = Field(1, ge=0, le=1)
    is_pool: int = Field(0, ge=0, le=1)
    is_main_road: int = Field(0, ge=0, le=1)
    is_garden_park: int = Field(1, ge=0, le=1)
    is_gated: int = Field(1, ge=0, le=1)
    is_corner: int = Field(0, ge=0, le=1)
    lat: float = Field(28.6139)
    lon: float = Field(77.2090)


class BuilderFloorRequest(SharedPredictRequest):
    pass


class ApartmentRequest(SharedPredictRequest):
    floor_level: str = Field("Medium (2nd - 7th)")
    is_ground: int = Field(0, ge=0, le=1)
    is_top: int = Field(0, ge=0, le=1)
    property_segment: str = Field("Mid")


class PlotRequest(BaseModel):
    area_sqft: float = Field(1800.0, gt=0)
    usage_type: str = Field("Residential")
    facing_direction: str = Field("North")
    circle_rate: float = Field(11891.59, gt=0)
    is_park_facing: int = Field(0, ge=0, le=1)
    is_corner: int = Field(0, ge=0, le=1)
    is_rectangular: int = Field(1, ge=0, le=1)
    is_gated: int = Field(1, ge=0, le=1)
    has_boundary_wall: int = Field(1, ge=0, le=1)
    road_width_upto_9m: int = Field(0, ge=0, le=1)
    road_width_9_to_18m: int = Field(1, ge=0, le=1)
    road_width_18_plus: int = Field(0, ge=0, le=1)
    lat: float = Field(28.6139)
    lon: float = Field(77.2090)


# ─────────────────────────────────────────────────────────────
# Parallel startup loading — all heavy I/O runs concurrently
# ─────────────────────────────────────────────────────────────

def _parallel_startup() -> dict:
    """Load all heavy resources in parallel using threads (I/O bound — GIL released)."""
    _startup_logger.info("Starting parallel resource load...")
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="startup") as pool:
        f_circle   = pool.submit(load_all_circle_rates)
        f_locality = pool.submit(load_plot_locality_encoding)
        f_road     = pool.submit(load_road_segments)
        f_ncr      = pool.submit(_compute_ncr_stats)
        f_bf       = pool.submit(load_bf_model)
        f_bf_rent  = pool.submit(load_bf_rent_model)
        f_apt      = pool.submit(load_apt_model)
        f_apt_rent = pool.submit(load_apt_rent_model)
        f_plot     = pool.submit(load_plot_model_bundle)
        f_bf_vor   = pool.submit(load_bf_voronoi)
        f_apt_vor  = pool.submit(load_apt_voronoi)

        result = {
            "circle_rates":      f_circle.result(),
            "locality_enc":      f_locality.result(),
            "road_segs":         f_road.result(),
            "ncr_stats":         f_ncr.result(),
            "bf":                f_bf.result(),
            "bf_rent":           f_bf_rent.result(),
            "apt":               f_apt.result(),
            "apt_rent":          f_apt_rent.result(),
            "plot_bundle":       f_plot.result(),
            "bf_vor":            f_bf_vor.result(),
            "apt_vor":           f_apt_vor.result(),
        }
    _startup_logger.info("Parallel resource load complete.")
    return result


_loaded = _parallel_startup()

circle_rates_by_city    = _loaded["circle_rates"]
locality_encoding_bundle = _loaded["locality_enc"]
road_segments            = _loaded["road_segs"]
NCR_STATS: dict[str, dict] = _loaded["ncr_stats"]
bf_model                 = _loaded["bf"]
bf_rent_model            = _loaded["bf_rent"]
apt_model                = _loaded["apt"]
apt_rent_model           = _loaded["apt_rent"]
plot_bundle              = _loaded["plot_bundle"]
plot_model               = plot_bundle.get("model")
bf_vor                   = _loaded["bf_vor"]
apt_vor                  = _loaded["apt_vor"]

# ForecastIntelligenceService is lightweight (no disk I/O in __init__)
forecast_service = ForecastIntelligenceService(project_root=".")

# MarketIntelligenceService: instantiate immediately (lightweight), build
# artifacts in a background daemon thread so startup is not blocked.
market_service = MarketIntelligenceService(project_root=str(PROJECT_ROOT))

def _mi_build_background() -> None:
    try:
        _any_exists = any(
            (PROJECT_ROOT / "opt" / seg / "market_intelligence.csv").exists()
            for seg in ("apt", "builder_floor", "plot")
        )
        if not _any_exists:
            _startup_logger.info("MarketIntelligenceService: building artifacts in background thread...")
            market_service.build_market_intelligence_artifacts()
            _startup_logger.info("MarketIntelligenceService: artifacts ready.")
    except Exception as _exc:
        _startup_logger.warning("MarketIntelligenceService background build failed (non-fatal): %s", _exc)

threading.Thread(target=_mi_build_background, daemon=True, name="mi-artifact-build").start()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/meta/model-registry")
def model_registry() -> dict[str, Any]:
    """Return the model version registry (written by the training pipeline)."""
    registry_path = Path("artifact/model_registry.json")
    if not registry_path.exists():
        return {"message": "No model registry found. Run the training pipeline first.", "registry": {}}
    with open(registry_path, "r") as f:
        return {"registry": json.load(f)}


@app.get("/meta/options")
def meta_options() -> dict[str, Any]:
    return {
        "cities": CITY_OPTIONS,
        "ageCategories": AGE_CATEGORIES,
        "furnishingCategories": FURNISHING_CATEGORIES,
        "facingCategories": FACING_CATEGORIES,
        "floorLevels": FLOOR_LEVELS,
        "aptPropertySegments": APT_PROPERTY_SEGMENTS,
        "plotUsageOptions": PLOT_USAGE_OPTIONS,
        "plotFacingOptions": PLOT_FACING_OPTIONS,
        "plotRoadWidthOptions": PLOT_ROAD_WIDTH_OPTIONS,
        "forecastSegments": forecast_service.available_segments(),
    }


@app.get("/meta/model-status")
def model_status() -> dict[str, Any]:
    return {
        "builderFloor": bf_model is not None,
        "builderFloorRent": bf_rent_model is not None,
        "apartment": apt_model is not None,
        "apartmentRent": apt_rent_model is not None,
        "plot": plot_model is not None,
        "plotModelPath": plot_bundle.get("model_path"),
        "plotFeatureCount": len(plot_bundle.get("features") or []),
    }


@app.get("/localities")
def localities(city: str = Query(...), query: str = Query(""), limit: int = Query(8, ge=1, le=20)) -> dict[str, list[str]]:
    suggestions = fuzzy_locality_suggestions(query, city, circle_rates_by_city, n=limit)
    return {"items": suggestions}


@app.get("/circle-rate")
def circle_rate(city: str = Query(...), locality: str = Query(...), property_type: str | None = Query(None)) -> dict[str, Any]:
    # Prefer CircleRateMatcher (loads from circle_rates/ at project root — correct path).
    # Fall back to the legacy lookup only if the matcher is unavailable.
    cr_matcher = getattr(market_service, "_cr_matcher", None)
    if cr_matcher is not None:
        value = cr_matcher.get_rate(city, locality)
    else:
        value = lookup_circle_rate(locality, city, circle_rates_by_city, property_type=property_type)
    return {"value": value}


@app.get("/locality-encoding")
def locality_encoding(locality: str = Query(...)) -> dict[str, float]:
    value = lookup_locality_target_encoding(locality, locality_encoding_bundle)
    return {"value": value}


@app.get("/geocode")
def geocode(city: str = Query(...), locality: str = Query(...)) -> dict[str, Any]:
    lat_lon = geocode_locality(locality, city)
    return {"latLon": lat_lon}


@app.post("/predict/builder-floor")
def predict_builder_floor(req: BuilderFloorRequest) -> dict[str, Any]:
    if bf_model is None:
        raise HTTPException(status_code=503, detail="Builder Floor model not available")

    vor_feats = compute_voronoi_features(req.lat, req.lon, bf_vor) if bf_vor is not None else {"voronoi_dist_to_seed": 0.0}

    X = build_features(
        bhk=req.bhk,
        area_sqft=req.area_sqft,
        bathrooms=req.bathrooms,
        balconies=req.balconies,
        circle_rate=req.circle_rate,
        is_parking=req.is_parking,
        is_pool=req.is_pool,
        is_main_road=req.is_main_road,
        is_garden_park=req.is_garden_park,
        is_gated=req.is_gated,
        is_corner=req.is_corner,
        age=req.age,
        furnishing=req.furnishing,
        facing=req.facing,
        voronoi_feats=vor_feats,
        model=bf_model,
    )

    pred_log = float(bf_model.predict(X)[0])
    pred_ratio = float(np.expm1(pred_log))
    ppsf = float(pred_ratio * req.circle_rate)
    total = float(ppsf * req.area_sqft)
    explanation = _build_xai_explanation(
        segment="builder-floor",
        ppsf=ppsf,
        circle_rate=req.circle_rate,
        area_sqft=req.area_sqft,
        pred_ratio=pred_ratio,
        voronoi_dist=float(vor_feats.get("voronoi_dist_to_seed", 5.0)),
        is_main_road=req.is_main_road,
        furnishing=req.furnishing,
        age=req.age,
        is_parking=req.is_parking,
        is_pool=req.is_pool,
        is_garden_park=req.is_garden_park,
        is_gated=req.is_gated,
        is_corner=req.is_corner,
        ncr_stats=NCR_STATS,
    )
    return {"predRatio": pred_ratio, "ppsf": ppsf, "total": total, "explanation": explanation}


@app.post("/predict/apartment")
def predict_apartment(req: ApartmentRequest) -> dict[str, Any]:
    if apt_model is None:
        raise HTTPException(status_code=503, detail="Apartment model not available")

    vor_feats = compute_voronoi_features(req.lat, req.lon, apt_vor) if apt_vor is not None else {"voronoi_dist_to_seed": 0.0}

    floor_feats = {
        "floor_low": 1 if "Low" in req.floor_level else 0,
        "floor_medium": 1 if "Medium" in req.floor_level else 0,
        "floor_high": 1 if "High" in req.floor_level else 0,
        "is_ground_floor": int(req.is_ground),
        "is_top_floor": int(req.is_top),
    }
    floor_feats.update(build_apartment_property_features(req.property_segment, apt_model))

    X = build_features(
        bhk=req.bhk,
        area_sqft=req.area_sqft,
        bathrooms=req.bathrooms,
        balconies=req.balconies,
        circle_rate=req.circle_rate,
        is_parking=req.is_parking,
        is_pool=req.is_pool,
        is_main_road=req.is_main_road,
        is_garden_park=req.is_garden_park,
        is_gated=req.is_gated,
        is_corner=req.is_corner,
        age=req.age,
        furnishing=req.furnishing,
        facing=req.facing,
        voronoi_feats=vor_feats,
        model=apt_model,
        floor_feats=floor_feats,
    )

    pred_log = float(apt_model.predict(X)[0])
    pred_ratio = float(np.expm1(pred_log))
    ppsf = float(pred_ratio * req.circle_rate)
    total = float(ppsf * req.area_sqft)
    explanation = _build_xai_explanation(
        segment="apartment",
        ppsf=ppsf,
        circle_rate=req.circle_rate,
        area_sqft=req.area_sqft,
        pred_ratio=pred_ratio,
        voronoi_dist=float(vor_feats.get("voronoi_dist_to_seed", 5.0)),
        is_main_road=req.is_main_road,
        furnishing=req.furnishing,
        age=req.age,
        floor_level=req.floor_level,
        property_segment=req.property_segment,
        is_parking=req.is_parking,
        is_pool=req.is_pool,
        is_garden_park=req.is_garden_park,
        is_gated=req.is_gated,
        is_corner=req.is_corner,
        ncr_stats=NCR_STATS,
    )
    return {"predRatio": pred_ratio, "ppsf": ppsf, "total": total, "explanation": explanation}


@app.post("/predict/plot")
def predict_plot(req: PlotRequest) -> dict[str, Any]:
    if plot_model is None:
        raise HTTPException(status_code=503, detail="Plot model not available")

    X = build_plot_features(
        area_sqft=req.area_sqft,
        circle_rate=req.circle_rate,
        latitude=req.lat,
        longitude=req.lon,
        usage_type=req.usage_type,
        facing_direction=req.facing_direction,
        is_park_facing=req.is_park_facing,
        is_corner=req.is_corner,
        is_rectangular=req.is_rectangular,
        is_gated=req.is_gated,
        has_boundary_wall=req.has_boundary_wall,
        road_width_upto_9m=req.road_width_upto_9m,
        road_width_9_to_18m=req.road_width_9_to_18m,
        road_width_18_plus=req.road_width_18_plus,
        model_bundle=plot_bundle,
        road_segments=road_segments,
    )

    pred_log_ppsf = float(plot_model.predict(X)[0])
    pred_ppsf = float(np.expm1(pred_log_ppsf))
    total = float(pred_ppsf * req.area_sqft)
    # Road distances for XAI (re-use already-loaded segments)
    _nh = nearest_distance_km(req.lat, req.lon, road_segments["NH"])
    _sh = nearest_distance_km(req.lat, req.lon, road_segments["SH"])
    _mdr = nearest_distance_km(req.lat, req.lon, road_segments["MDR"])
    explanation = _build_xai_explanation(
        segment="plot",
        ppsf=pred_ppsf,
        circle_rate=req.circle_rate,
        area_sqft=req.area_sqft,
        nh_km=0.0 if np.isnan(_nh) else float(_nh),
        sh_km=0.0 if np.isnan(_sh) else float(_sh),
        mdr_km=0.0 if np.isnan(_mdr) else float(_mdr),
        road_width_18_plus=req.road_width_18_plus,
        road_width_9_to_18m=req.road_width_9_to_18m,
        usage_type=req.usage_type,
        is_gated=req.is_gated,
        has_boundary_wall=req.has_boundary_wall,
        is_park_facing=req.is_park_facing,
        is_corner=req.is_corner,
        is_rectangular=req.is_rectangular,
        ncr_stats=NCR_STATS,
    )
    return {"ppsf": pred_ppsf, "total": total, "explanation": explanation}


@app.post("/predict/builder-floor/all")
def predict_builder_floor_all(req: BuilderFloorRequest) -> dict[str, Any]:
    """Sell price + rent estimate + rental yield in one call for Builder Floor."""
    if bf_model is None:
        raise HTTPException(status_code=503, detail="Builder Floor model not available")

    vor_feats = compute_voronoi_features(req.lat, req.lon, bf_vor) if bf_vor is not None else {"voronoi_dist_to_seed": 0.0}

    X_sell = build_features(
        bhk=req.bhk, area_sqft=req.area_sqft, bathrooms=req.bathrooms, balconies=req.balconies,
        circle_rate=req.circle_rate, is_parking=req.is_parking, is_pool=req.is_pool,
        is_main_road=req.is_main_road, is_garden_park=req.is_garden_park,
        is_gated=req.is_gated, is_corner=req.is_corner,
        age=req.age, furnishing=req.furnishing, facing=req.facing,
        voronoi_feats=vor_feats, model=bf_model,
    )
    pred_log = float(bf_model.predict(X_sell)[0])
    pred_ratio = float(np.expm1(pred_log))
    ppsf = float(pred_ratio * req.circle_rate)
    total = float(ppsf * req.area_sqft)

    monthly_rent: float | None = None
    annual_rent: float | None = None
    rent_yield_pct: float | None = None
    rent_available = bf_rent_model is not None

    if rent_available:
        try:
            floor_feats_rent = {
                "current_floor": 1,
                "total_floors": max(2, req.bhk + 1),
                "is_ground_floor": 0,
                "is_top_floor": 0,
                "is_basement": 0,
            }
            X_rent = build_features(
                bhk=req.bhk, area_sqft=req.area_sqft, bathrooms=req.bathrooms, balconies=req.balconies,
                circle_rate=req.circle_rate, is_parking=req.is_parking, is_pool=req.is_pool,
                is_main_road=req.is_main_road, is_garden_park=req.is_garden_park,
                is_gated=req.is_gated, is_corner=req.is_corner,
                age=req.age, furnishing=req.furnishing, facing=req.facing,
                voronoi_feats=vor_feats, model=bf_rent_model, floor_feats=floor_feats_rent,
            )
            rent_log = float(bf_rent_model.predict(X_rent)[0])
            monthly_rent = max(0.0, float(np.expm1(rent_log)))
            annual_rent = monthly_rent * 12.0
            rent_yield_pct = round((annual_rent / total) * 100.0, 3) if total > 0 else None
        except Exception:
            rent_available = False

    explanation = _build_xai_explanation(
        segment="builder-floor", ppsf=ppsf, circle_rate=req.circle_rate, area_sqft=req.area_sqft,
        pred_ratio=pred_ratio, voronoi_dist=float(vor_feats.get("voronoi_dist_to_seed", 5.0)),
        is_main_road=req.is_main_road, furnishing=req.furnishing, age=req.age,
        is_parking=req.is_parking, is_pool=req.is_pool, is_garden_park=req.is_garden_park,
        is_gated=req.is_gated, is_corner=req.is_corner, ncr_stats=NCR_STATS,
    )
    return {
        "segment": "builder-floor",
        "sell": {"predRatio": pred_ratio, "ppsf": round(ppsf, 2), "total": round(total, 2)},
        "rent": {
            "available": rent_available,
            "monthlyRent": round(monthly_rent, 2) if monthly_rent is not None else None,
            "annualRent": round(annual_rent, 2) if annual_rent is not None else None,
            "rentalYieldPct": rent_yield_pct,
        },
        "explanation": explanation,
    }


@app.post("/predict/apartment/all")
def predict_apartment_all(req: ApartmentRequest) -> dict[str, Any]:
    """Sell price + rent estimate + rental yield in one call for Apartment."""
    if apt_model is None:
        raise HTTPException(status_code=503, detail="Apartment model not available")

    vor_feats = compute_voronoi_features(req.lat, req.lon, apt_vor) if apt_vor is not None else {"voronoi_dist_to_seed": 0.0}

    floor_feats_sell = {
        "floor_low": 1 if "Low" in req.floor_level else 0,
        "floor_medium": 1 if "Medium" in req.floor_level else 0,
        "floor_high": 1 if "High" in req.floor_level else 0,
        "is_ground_floor": int(req.is_ground),
        "is_top_floor": int(req.is_top),
    }
    floor_feats_sell.update(build_apartment_property_features(req.property_segment, apt_model))

    X_sell = build_features(
        bhk=req.bhk, area_sqft=req.area_sqft, bathrooms=req.bathrooms, balconies=req.balconies,
        circle_rate=req.circle_rate, is_parking=req.is_parking, is_pool=req.is_pool,
        is_main_road=req.is_main_road, is_garden_park=req.is_garden_park,
        is_gated=req.is_gated, is_corner=req.is_corner,
        age=req.age, furnishing=req.furnishing, facing=req.facing,
        voronoi_feats=vor_feats, model=apt_model, floor_feats=floor_feats_sell,
    )
    pred_log = float(apt_model.predict(X_sell)[0])
    pred_ratio = float(np.expm1(pred_log))
    ppsf = float(pred_ratio * req.circle_rate)
    total = float(ppsf * req.area_sqft)

    monthly_rent: float | None = None
    annual_rent: float | None = None
    rent_yield_pct: float | None = None
    rent_available = apt_rent_model is not None

    if rent_available:
        try:
            floor_feats_rent = {
                "floor_low": 0, "floor_medium": 1, "floor_high": 0,
                "is_ground_floor": 0, "is_top_floor": 0,
            }
            floor_feats_rent.update(build_apartment_property_features("Mid", apt_rent_model))
            X_rent = build_features(
                bhk=req.bhk, area_sqft=req.area_sqft, bathrooms=req.bathrooms, balconies=req.balconies,
                circle_rate=req.circle_rate, is_parking=req.is_parking, is_pool=req.is_pool,
                is_main_road=req.is_main_road, is_garden_park=req.is_garden_park,
                is_gated=req.is_gated, is_corner=req.is_corner,
                age=req.age, furnishing=req.furnishing, facing=req.facing,
                voronoi_feats=vor_feats, model=apt_rent_model, floor_feats=floor_feats_rent,
            )
            rent_log = float(apt_rent_model.predict(X_rent)[0])
            monthly_rent = max(0.0, float(np.expm1(rent_log)))
            annual_rent = monthly_rent * 12.0
            rent_yield_pct = round((annual_rent / total) * 100.0, 3) if total > 0 else None
        except Exception:
            rent_available = False

    explanation = _build_xai_explanation(
        segment="apartment", ppsf=ppsf, circle_rate=req.circle_rate, area_sqft=req.area_sqft,
        pred_ratio=pred_ratio, voronoi_dist=float(vor_feats.get("voronoi_dist_to_seed", 5.0)),
        is_main_road=req.is_main_road, furnishing=req.furnishing, age=req.age,
        floor_level=req.floor_level, property_segment=req.property_segment,
        is_parking=req.is_parking, is_pool=req.is_pool, is_garden_park=req.is_garden_park,
        is_gated=req.is_gated, is_corner=req.is_corner, ncr_stats=NCR_STATS,
    )
    return {
        "segment": "apartment",
        "sell": {"predRatio": pred_ratio, "ppsf": round(ppsf, 2), "total": round(total, 2)},
        "rent": {
            "available": rent_available,
            "monthlyRent": round(monthly_rent, 2) if monthly_rent is not None else None,
            "annualRent": round(annual_rent, 2) if annual_rent is not None else None,
            "rentalYieldPct": rent_yield_pct,
        },
        "explanation": explanation,
    }


@app.get("/road-distances")
def get_road_distances(lat: float = Query(...), lon: float = Query(...)) -> dict[str, float]:
    """Calculate distances to MDR, SH, and NH roads from given coordinates."""
    try:
        mdr_km = nearest_distance_km(float(lat), float(lon), road_segments["MDR"])
        sh_km = nearest_distance_km(float(lat), float(lon), road_segments["SH"])
        nh_km = nearest_distance_km(float(lat), float(lon), road_segments["NH"])
        
        return {
            "closest_distance_MDR_km": 0.0 if np.isnan(mdr_km) else float(round(mdr_km, 2)),
            "closest_distance_SH_km": 0.0 if np.isnan(sh_km) else float(round(sh_km, 2)),
            "closest_distance_NH_km": 0.0 if np.isnan(nh_km) else float(round(nh_km, 2)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate road distances: {str(e)}")


@app.get("/forecast/segments")
def forecast_segments() -> dict[str, Any]:
    return {"items": forecast_service.available_segments()}


@app.get("/forecast/cities")
def forecast_cities(segment: str = Query(...)) -> dict[str, Any]:
    try:
        items = forecast_service.list_cities(segment=segment)
        return {"items": items}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/forecast/localities")
def forecast_localities(
    segment: str = Query(...),
    city: str = Query(""),
    query: str = Query(""),
    limit: int = Query(8, ge=1, le=30),
) -> dict[str, Any]:
    try:
        items = forecast_service.suggest_localities(segment=segment, city=city, query=query, limit=limit)
        return {"items": items}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/forecast/property-ids")
def forecast_property_ids(
    segment: str = Query(...),
    locality: str = Query(...),
    city: str = Query(""),
    limit: int = Query(250, ge=1, le=1000),
) -> dict[str, Any]:
    try:
        items = forecast_service.list_property_ids(segment=segment, locality=locality, city=city, limit=limit)
        return {"items": items}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/forecast/overview")
def forecast_overview(segment: str = Query(...)) -> dict[str, Any]:
    try:
        return forecast_service.overview(segment=segment)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/forecast/context")
def forecast_context(
    segment: str = Query(...),
    city: str = Query(""),
    locality: str = Query(...),
    property_id: str | None = Query(None),
    years: int = Query(5, ge=1, le=10),
) -> dict[str, Any]:
    try:
        return forecast_service.context(
            segment=segment,
            city=city,
            locality=locality,
            property_id=property_id,
            years=years,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build forecast context: {exc}")


@app.get("/insights/buy-decision")
def insights_buy_decision(
    segment: str = Query(...),
    city: str = Query(""),
    locality: str = Query(...),
    property_id: str | None = Query(None),
    hold_years: int = Query(5, ge=1, le=10),
) -> dict[str, Any]:
    try:
        ctx_years = max(hold_years, 5)
        context_payload = forecast_service.context(
            segment=segment,
            city=city,
            locality=locality,
            property_id=property_id,
            years=ctx_years,
        )
        decision = _build_buy_decision_payload(context_payload, hold_years=hold_years)
        return {
            "segment": segment,
            "city": city,
            "locality": locality,
            "selectedPropertyId": context_payload.get("selectedPropertyId"),
            "decision": decision,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to compute buy decision: {exc}")


@app.get("/insights/roi")
def insights_roi(
    segment: str = Query(...),
    city: str = Query(""),
    locality: str = Query(...),
    property_id: str | None = Query(None),
    hold_years: int = Query(5, ge=1, le=10),
    area_sqft: float | None = Query(None, gt=0),
    purchase_cost_pct: float = Query(7.0, ge=0, le=20),
    annual_holding_cost_pct: float = Query(1.5, ge=0, le=10),
    exit_cost_pct: float = Query(2.0, ge=0, le=10),
    rent_yield_pct: float | None = Query(None, ge=0, le=10),
) -> dict[str, Any]:
    try:
        ctx_years = max(hold_years, 5)
        context_payload = forecast_service.context(
            segment=segment,
            city=city,
            locality=locality,
            property_id=property_id,
            years=ctx_years,
        )
        roi_payload = _build_roi_payload(
            context_payload,
            hold_years=hold_years,
            area_sqft_override=area_sqft,
            purchase_cost_pct=purchase_cost_pct,
            annual_holding_cost_pct=annual_holding_cost_pct,
            exit_cost_pct=exit_cost_pct,
            rent_yield_pct=rent_yield_pct,
        )
        return {
            "segment": segment,
            "city": city,
            "locality": locality,
            "selectedPropertyId": context_payload.get("selectedPropertyId"),
            **roi_payload,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to compute ROI: {exc}")


# ─────────────────────────────────────────────────────────────
# Market Intelligence endpoints
# ─────────────────────────────────────────────────────────────

_MI_SEGMENT_LABELS = {
    "apt": "Apartment",
    "builder_floor": "Builder Floor",
    "plot": "Plot",
}


@app.get("/market-intelligence/segments")
def mi_segments() -> dict[str, Any]:
    available: list[dict[str, Any]] = []
    for seg_id, seg_label in _MI_SEGMENT_LABELS.items():
        from pathlib import Path as _Path
        artifact_exists = (_Path(str(PROJECT_ROOT)) / "opt" / seg_id / "market_intelligence.csv").exists()
        available.append({"id": seg_id, "label": seg_label, "available": artifact_exists})
    return {"segments": available}


@app.get("/market-intelligence/cities")
def mi_cities(segment: str = Query(...)) -> dict[str, Any]:
    try:
        cities = market_service.get_available_cities(segment)
        return {"segment": segment, "cities": cities}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Market intelligence cities failed: {exc}")


@app.get("/market-intelligence/localities")
def mi_localities(
    segment: str = Query(...),
    city: str = Query(...),
    query: str = Query(""),
    limit: int = Query(30, ge=1, le=100),
) -> dict[str, Any]:
    try:
        localities = market_service.get_available_localities(segment, city, query=query, limit=limit)
        return {"segment": segment, "city": city, "localities": localities}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Market intelligence localities failed: {exc}")


@app.get("/market-intelligence/context")
def mi_context(
    segment: str = Query(...),
    city: str = Query(...),
    locality: str = Query(...),
) -> dict[str, Any]:
    try:
        result = market_service.get_market_context(segment, city, locality)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Market intelligence context failed: {exc}")


# ─────────────────────────────────────────────────────────────
# Metro data (loaded once at startup)
# ─────────────────────────────────────────────────────────────

def _load_metro_stations() -> np.ndarray:
    """Load Delhi metro station lat/lon as (N, 2) numpy array."""
    candidates = [
        PROJECT_ROOT / "DELHI_METRO_DATA (1).csv",
        PROJECT_ROOT / "DELHI_METRO_DATA.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                lats = pd.to_numeric(df.get("Latitude"), errors="coerce")
                lons = pd.to_numeric(df.get("Longitude"), errors="coerce")
                mask = lats.notna() & lons.notna()
                arr = np.column_stack([lats[mask].values, lons[mask].values])
                _startup_logger.info("Loaded %d metro stations from %s", len(arr), path.name)
                return arr
            except Exception as exc:
                _startup_logger.warning("Failed to load metro data from %s: %s", path, exc)
    _startup_logger.warning("Metro data file not found; metro distances will be unavailable.")
    return np.empty((0, 2), dtype=float)


METRO_STATIONS: np.ndarray = _load_metro_stations()


def _nearest_metro_km(lat: float, lon: float) -> float:
    """Haversine distance (km) to the nearest metro station."""
    if METRO_STATIONS.shape[0] == 0:
        return float("nan")
    k_lat = 110.574
    k_lon = 111.320 * np.cos(np.radians(lat))
    dy = (METRO_STATIONS[:, 0] - lat) * k_lat
    dx = (METRO_STATIONS[:, 1] - lon) * k_lon
    dists = np.sqrt(dx ** 2 + dy ** 2)
    return float(np.nanmin(dists))


# ─────────────────────────────────────────────────────────────
# Property Intelligence – per-listing analysis
# ─────────────────────────────────────────────────────────────

class PIBfRequest(SharedPredictRequest):
    listing_price: float = Field(..., gt=0, description="Seller's asking total price in INR")
    locality: str = Field("", description="Locality name for market context lookup")
    city: str = Field("Delhi", description="City")


class PIAptRequest(PIBfRequest):
    floor_level: str = Field("Medium (2nd - 7th)")
    is_ground: int = Field(0, ge=0, le=1)
    is_top: int = Field(0, ge=0, le=1)
    property_segment: str = Field("Mid")


class PIPlotRequest(BaseModel):
    listing_price: float = Field(..., gt=0)
    locality: str = Field("")
    city: str = Field("Delhi")
    area_sqft: float = Field(1800.0, gt=0)
    usage_type: str = Field("Residential")
    facing_direction: str = Field("North")
    circle_rate: float = Field(11891.59, gt=0)
    is_park_facing: int = Field(0, ge=0, le=1)
    is_corner: int = Field(0, ge=0, le=1)
    is_rectangular: int = Field(1, ge=0, le=1)
    is_gated: int = Field(1, ge=0, le=1)
    has_boundary_wall: int = Field(1, ge=0, le=1)
    road_width_upto_9m: int = Field(0, ge=0, le=1)
    road_width_9_to_18m: int = Field(1, ge=0, le=1)
    road_width_18_plus: int = Field(0, ge=0, le=1)
    lat: float = Field(28.6139)
    lon: float = Field(77.2090)


def _pi_score(value: float, good_threshold: float, bad_threshold: float, good_is_low: bool = True) -> int:
    """
    Return a score in {-2, -1, 0, 1, 2}.
    If good_is_low: lower value is better (e.g. distance to metro).
    Otherwise: higher value is better.
    """
    if good_is_low:
        if value <= good_threshold:
            return 2
        if value <= (good_threshold + bad_threshold) / 2:
            return 1
        if value <= bad_threshold:
            return 0
        if value <= bad_threshold * 1.5:
            return -1
        return -2
    else:
        if value >= good_threshold:
            return 2
        if value >= (good_threshold + bad_threshold) / 2:
            return 1
        if value >= bad_threshold:
            return 0
        if value >= bad_threshold * 0.5:
            return -1
        return -2


_SCORE_LABELS = {2: "Strong +", 1: "Positive", 0: "Neutral", -1: "Negative", -2: "Strong −"}


def _build_property_intelligence(
    *,
    segment: str,
    listing_price: float,
    fair_value_ppsf: float,
    fair_value_total: float,
    area_sqft: float,
    circle_rate: float,
    lat: float,
    lon: float,
    locality: str,
    city: str,
    # shared BF / APT fields (also used for rent model)
    bhk: int = 2,
    bathrooms: int = 2,
    balconies: int = 1,
    age: str = "5 to 10 years",
    is_gated: int = 0,
    is_parking: int = 0,
    is_pool: int = 0,
    is_garden_park: int = 0,
    is_main_road: int = 0,
    is_corner: int = 0,
    furnishing: str = "Semi-Furnished",
    facing: str = "North",
    floor_level: str = "Medium (2nd - 7th)",
    property_segment: str = "Mid",
    # plot signals
    is_rectangular: int = 1,
    has_boundary_wall: int = 0,
    is_park_facing: int = 0,
    road_width_18_plus: int = 0,
    road_width_9_to_18m: int = 0,
    usage_type: str = "Residential",
    # voronoi / uniqueness
    voronoi_dist: float = 5.0,
    model_confidence: float = 0.8,
    ncr_stats: dict | None = None,
) -> dict[str, Any]:
    """Build all 12 property intelligence signals for a single listing."""

    listing_ppsf = listing_price / max(area_sqft, 1.0)
    premium_discount_pct = (listing_ppsf / max(fair_value_ppsf, 1.0) - 1.0) * 100.0

    # 1. Listing premium / discount
    pd_score = _pi_score(-premium_discount_pct, 10, -10, good_is_low=False)  # discount is good
    pd_signal = {
        "key": "listing_premium_discount",
        "label": "Listing Premium / Discount",
        "score": pd_score,
        "scoreLabel": _SCORE_LABELS[pd_score],
        "value": round(premium_discount_pct, 2),
        "unit": "%",
        "detail": (
            f"Listing price (INR {listing_ppsf:,.0f}/sqft) is "
            f"{'above' if premium_discount_pct > 0 else 'below'} fair value "
            f"(INR {fair_value_ppsf:,.0f}/sqft) by {abs(premium_discount_pct):.1f}%. "
            + ("Potential discount — attractive entry." if premium_discount_pct < -5 else
               "Near fair value." if abs(premium_discount_pct) <= 5 else
               "Premium pricing — check if quality justifies the gap.")
        ),
    }

    # 2. Uniqueness (based on voronoi distance from nearest cluster seed)
    uniqueness = min(1.0, voronoi_dist / 10.0)
    uniq_score = _pi_score(uniqueness, 0.7, 0.3, good_is_low=False)
    uniq_signal = {
        "key": "uniqueness",
        "label": "Uniqueness",
        "score": uniq_score,
        "scoreLabel": _SCORE_LABELS[uniq_score],
        "value": round(uniqueness, 3),
        "unit": "index (0–1)",
        "detail": (
            f"Property uniqueness index: {uniqueness:.2f} (voronoi dist {voronoi_dist:.1f} km). "
            + ("Highly differentiated from nearby substitutes." if uniqueness >= 0.7 else
               "Moderate differentiation." if uniqueness >= 0.3 else
               "Many similar competing properties nearby.")
        ),
    }

    # 3. Model confidence
    conf_score = _pi_score(model_confidence, 0.85, 0.6, good_is_low=False)
    conf_signal = {
        "key": "model_confidence",
        "label": "Valuation Confidence",
        "score": conf_score,
        "scoreLabel": _SCORE_LABELS[conf_score],
        "value": round(model_confidence * 100, 1),
        "unit": "%",
        "detail": (
            f"Model confidence: {model_confidence*100:.0f}%. "
            + ("High certainty — valuation is reliable." if model_confidence >= 0.85 else
               "Moderate confidence." if model_confidence >= 0.6 else
               "Low confidence — treat valuation as indicative only.")
        ),
    }

    # 4. Undervaluation relative to confidence (composite)
    underval_ratio = max(0.0, -premium_discount_pct) * model_confidence / 100.0
    uv_score = _pi_score(underval_ratio, 0.1, 0.02, good_is_low=False)
    uv_signal = {
        "key": "undervaluation_confidence",
        "label": "Undervaluation × Confidence",
        "score": uv_score,
        "scoreLabel": _SCORE_LABELS[uv_score],
        "value": round(underval_ratio, 3),
        "unit": "composite",
        "detail": (
            f"Discount ({abs(premium_discount_pct):.1f}%) × confidence ({model_confidence*100:.0f}%) = {underval_ratio:.3f}. "
            + ("Strong buy signal — confident undervaluation." if uv_score >= 1 else
               "Mild undervaluation or low confidence." if uv_score == 0 else
               "No clear undervaluation advantage.")
        ),
    }

    # 5. Metro distance
    metro_km = _nearest_metro_km(lat, lon)
    metro_valid = not np.isnan(metro_km)
    metro_score = _pi_score(metro_km, 1.0, 3.0, good_is_low=True) if metro_valid else 0
    metro_signal = {
        "key": "metro_distance",
        "label": "Distance to Metro",
        "score": metro_score,
        "scoreLabel": _SCORE_LABELS[metro_score] if metro_valid else "N/A",
        "value": round(metro_km, 2) if metro_valid else None,
        "unit": "km",
        "detail": (
            f"Nearest metro station: {metro_km:.2f} km. "
            + ("Excellent metro access (< 1 km)." if metro_valid and metro_km < 1.0 else
               "Good metro access (1–3 km)." if metro_valid and metro_km < 3.0 else
               "Moderate connectivity (3–5 km)." if metro_valid and metro_km < 5.0 else
               "Poor metro access (> 5 km)." if metro_valid else
               "Metro distance data unavailable.")
        ) if metro_valid else "Metro distance data unavailable.",
    }

    # 6. Safety proxies (from property features)
    safety_score_raw = int(is_gated) * 2 + int(is_main_road == 0) + int(has_boundary_wall) + int(is_park_facing)
    safety_max = 5
    safety_norm = safety_score_raw / safety_max
    safety_score = _pi_score(safety_norm, 0.7, 0.4, good_is_low=False)
    safety_factors = []
    if is_gated:
        safety_factors.append("gated community")
    if not is_main_road:
        safety_factors.append("away from main road noise")
    if has_boundary_wall:
        safety_factors.append("boundary wall")
    if is_park_facing:
        safety_factors.append("park-facing (open space)")
    safety_signal = {
        "key": "safety_proxies",
        "label": "Safety & Security Proxies",
        "score": safety_score,
        "scoreLabel": _SCORE_LABELS[safety_score],
        "value": round(safety_norm * 100, 1),
        "unit": "/ 100",
        "detail": (
            ("Positive safety features: " + ", ".join(safety_factors) + "." if safety_factors else "No notable safety features detected.")
        ),
    }

    # 7. Unit desirability (floor, facing, furnishing, parking, balcony, pool)
    desirability_pts = 0
    desirability_max = 8
    desirability_notes = []

    facing_premium = {"East": 2, "North": 2, "North-East": 2, "North-West": 1, "South-East": 1}
    fp = facing_premium.get(facing, 0)
    desirability_pts += fp
    desirability_max += 2
    if fp >= 2:
        desirability_notes.append(f"{facing} facing (premium)")
    elif fp == 1:
        desirability_notes.append(f"{facing} facing (good)")

    if furnishing == "Furnished":
        desirability_pts += 2
        desirability_notes.append("Fully furnished")
    elif furnishing == "Semi-Furnished":
        desirability_pts += 1
        desirability_notes.append("Semi-furnished")

    if is_parking:
        desirability_pts += 2
        desirability_notes.append("Parking")
    if is_pool:
        desirability_pts += 1
        desirability_notes.append("Pool")
    if is_garden_park:
        desirability_pts += 1
        desirability_notes.append("Garden/Park")
    if is_corner:
        desirability_pts += 1
        desirability_notes.append("Corner property")

    if "High" in floor_level:
        desirability_pts += 2
        desirability_notes.append("High floor")
    elif "Medium" in floor_level:
        desirability_pts += 1
        desirability_notes.append("Mid floor")

    # plot specific
    if is_rectangular:
        desirability_pts += 1
        desirability_notes.append("Rectangular shape")
    if road_width_18_plus:
        desirability_pts += 2
        desirability_notes.append("Wide road (18m+)")
    elif road_width_9_to_18m:
        desirability_pts += 1
        desirability_notes.append("Medium road (9–18m)")

    desirability_norm = min(1.0, desirability_pts / max(desirability_max, 1))
    des_score = _pi_score(desirability_norm, 0.6, 0.35, good_is_low=False)
    desirability_signal = {
        "key": "unit_desirability",
        "label": "Unit Desirability",
        "score": des_score,
        "scoreLabel": _SCORE_LABELS[des_score],
        "value": round(desirability_norm * 100, 1),
        "unit": "/ 100",
        "detail": (", ".join(desirability_notes) + "." if desirability_notes else "Standard unit with no notable premium features."),
    }

    # 8–12: Locality-level market signals from MarketIntelligenceService
    seg_map = {"builder-floor": "builder_floor", "apartment": "apt", "plot": "plot"}
    mi_seg = seg_map.get(segment.lower().replace("_", "-"), "apt")
    mi_data: dict[str, Any] = {}
    if locality.strip():
        try:
            mi_data = market_service.get_market_context(mi_seg, city, locality) or {}
        except Exception:
            mi_data = {}

    kpis = mi_data.get("kpis", {})
    idx = mi_data.get("indices", {})

    # 8. Inventory pressure — competing supply of the SAME segment in this locality.
    # A builder-floor buyer competes with other builder floors, not apartments or plots.
    # Cap at 50 for locality-level scoring (200 is citywide scale, not per-locality).
    supply_stock = kpis.get("activeSupplyStock", None)
    stale_share = kpis.get("staleInventoryShare", 0.0)
    if supply_stock is not None and supply_stock > 0:
        inv_norm = min(1.0, supply_stock / 50.0)
        inv_combined = (inv_norm * 0.6 + stale_share * 0.4)
        inv_score = _pi_score(1.0 - inv_combined, 0.7, 0.4, good_is_low=False)
        inv_detail = f"Active supply: {supply_stock} {mi_seg} listings; stale inventory: {stale_share*100:.1f}%."
    else:
        inv_score = 0
        inv_detail = "Inventory data unavailable for this locality."
    inv_signal = {
        "key": "inventory_pressure",
        "label": "Inventory Pressure",
        "score": inv_score,
        "scoreLabel": _SCORE_LABELS[inv_score],
        "value": supply_stock,
        "unit": "active listings",
        "detail": inv_detail,
    }

    # 9. Price cut frequency
    cut_freq = kpis.get("priceCutFrequency", None)
    if cut_freq is not None:
        cut_score = _pi_score(1.0 - cut_freq, 0.85, 0.6, good_is_low=False)
        cut_detail = f"Price-cut frequency: {cut_freq*100:.1f}% of updated listings reduced price."
    else:
        cut_score = 0
        cut_detail = "Price cut data unavailable for this locality."
    cut_signal = {
        "key": "price_cut_frequency",
        "label": "Price Cut Frequency",
        "score": cut_score,
        "scoreLabel": _SCORE_LABELS[cut_score],
        "value": round(cut_freq * 100, 1) if cut_freq is not None else None,
        "unit": "%",
        "detail": cut_detail,
    }

    # 10. Neighbourhood liquidity
    liq_idx = idx.get("liquidityIndex", None)
    absorption = kpis.get("absorptionRate", None)
    if liq_idx is not None:
        liq_score = _pi_score(liq_idx, 65, 40, good_is_low=False)
        liq_detail = f"Liquidity index: {liq_idx:.1f}/100."
        if absorption is not None:
            liq_detail += f" Absorption proxy: {absorption*100:.1f}%."
    else:
        liq_score = 0
        liq_detail = "Liquidity data unavailable for this locality."
    liq_signal = {
        "key": "neighbourhood_liquidity",
        "label": "Neighbourhood Liquidity",
        "score": liq_score,
        "scoreLabel": _SCORE_LABELS[liq_score],
        "value": round(liq_idx, 1) if liq_idx is not None else None,
        "unit": "/ 100",
        "detail": liq_detail,
    }

    # 11. Rental yield — use actual rent model when available
    seg_norm = segment.strip().lower().replace("_", "-")
    monthly_rent_estimate: float | None = None
    rent_yield_source = "benchmark"
    est_yield_pct: float
    yield_benchmark = {"apartment": 3.5, "builder-floor": 3.0, "plot": 1.5}

    if seg_norm in {"builder-floor", "apartment"}:
        rent_model = bf_rent_model if seg_norm == "builder-floor" else apt_rent_model
        try:
            if rent_model is not None:
                common_kwargs = {
                    "bhk": bhk,
                    "area_sqft": area_sqft,
                    "bathrooms": bathrooms,
                    "balconies": balconies,
                    "circle_rate": circle_rate,
                    "is_parking": is_parking,
                    "is_pool": is_pool,
                    "is_main_road": is_main_road,
                    "is_garden_park": is_garden_park,
                    "is_gated": is_gated,
                    "is_corner": is_corner,
                    "age": age,
                    "furnishing": furnishing,
                    "facing": facing,
                    "voronoi_feats": {"voronoi_dist_to_seed": voronoi_dist},
                }
                if seg_norm == "builder-floor":
                    floor_feats = {
                        "current_floor": 1,
                        "total_floors": max(2, bhk + 1),
                        "is_ground_floor": 0,
                        "is_top_floor": 0,
                        "is_basement": 0,
                    }
                else:
                    floor_feats = {
                        "floor_low": 1 if "Low" in floor_level else 0,
                        "floor_medium": 1 if "Medium" in floor_level else 0,
                        "floor_high": 1 if "High" in floor_level else 0,
                        "is_ground_floor": 0,
                        "is_top_floor": 0,
                    }
                    floor_feats.update(build_apartment_property_features(property_segment, rent_model))
                rent_X = build_features(model=rent_model, floor_feats=floor_feats, **common_kwargs)
                rent_pred_log = float(rent_model.predict(rent_X)[0])
                monthly_rent_estimate = max(0.0, float(np.expm1(rent_pred_log)))
                annual_rent = monthly_rent_estimate * 12.0
                est_yield_pct = (annual_rent / max(listing_price, 1.0)) * 100.0
                rent_yield_source = "rent_model"
            else:
                raise ValueError("no rent model")
        except Exception:
            # Fallback to benchmark adjusted by price premium vs median
            stats = (ncr_stats or {}).get(seg_norm, {})
            median_ppsf = stats.get("median_ppsf", fair_value_ppsf)
            est_yield_pct = yield_benchmark.get(seg_norm, 3.0)
            if median_ppsf > 0:
                est_yield_pct = est_yield_pct / max(listing_ppsf / median_ppsf, 0.5)
            rent_yield_source = "benchmark"
    else:
        # Plot: no rent model — use benchmark adjusted by listing vs median
        stats = (ncr_stats or {}).get(seg_norm, {})
        median_ppsf = stats.get("median_ppsf", fair_value_ppsf)
        est_yield_pct = yield_benchmark.get("plot", 1.5)
        if median_ppsf > 0:
            est_yield_pct = est_yield_pct / max(listing_ppsf / median_ppsf, 0.5)
        rent_yield_source = "benchmark"

    yield_score = _pi_score(est_yield_pct, 3.5, 2.0, good_is_low=False)
    yield_detail = f"Estimated gross rental yield: {est_yield_pct:.2f}% p.a."
    if rent_yield_source == "rent_model" and monthly_rent_estimate is not None:
        yield_detail += f" — rent model predicted INR {monthly_rent_estimate:,.0f}/month"
    elif rent_yield_source == "benchmark":
        yield_detail += f" (segment benchmark; rent model unavailable)"
    yield_detail += ". " + (
        "Investor-attractive yield." if est_yield_pct >= 3.5 else
        "Moderate yield." if est_yield_pct >= 2.5 else
        "Yield may not support investment-grade return."
    )
    yield_signal = {
        "key": "rental_yield",
        "label": "Estimated Rental Yield",
        "score": yield_score,
        "scoreLabel": _SCORE_LABELS[yield_score],
        "value": round(est_yield_pct, 2),
        "unit": "% p.a.",
        "monthlyRentEstimate": round(monthly_rent_estimate, 0) if monthly_rent_estimate is not None else None,
        "source": rent_yield_source,
        "detail": yield_detail,
    }

    # 12. Expected growth (market heat + price momentum as forward proxy)
    heat = idx.get("marketHeatIndex", None)
    momentum = idx.get("priceMomentumScore", None)
    if heat is not None and momentum is not None:
        growth_score_raw = (heat * 0.5 + momentum * 0.5) / 100.0
        growth_score = _pi_score(growth_score_raw, 0.65, 0.4, good_is_low=False)
        growth_detail = f"Market heat: {heat:.1f}/100; price momentum: {momentum:.1f}/100."
    elif heat is not None:
        growth_score = _pi_score(heat, 65, 40, good_is_low=False)
        growth_detail = f"Market heat index: {heat:.1f}/100."
    else:
        growth_score = 0
        growth_detail = "Market growth signals unavailable for this locality — use locality-level Market Intelligence tab for more detail."
    growth_signal = {
        "key": "expected_growth",
        "label": "Expected Growth Potential",
        "score": growth_score,
        "scoreLabel": _SCORE_LABELS[growth_score],
        "value": round(heat, 1) if heat is not None else None,
        "unit": "/ 100 (heat)",
        "detail": growth_detail,
    }

    signals = [
        pd_signal, uniq_signal, conf_signal, uv_signal, metro_signal,
        safety_signal, desirability_signal, inv_signal, cut_signal, liq_signal,
        yield_signal, growth_signal,
    ]

    # Overall recommendation score
    total_score = sum(s["score"] for s in signals)
    max_possible = len(signals) * 2
    overall_pct = (total_score + max_possible) / (2 * max_possible) * 100

    if overall_pct >= 70:
        recommendation = "Strong Buy"
        rec_color = "#2a7d4f"
    elif overall_pct >= 58:
        recommendation = "Buy"
        rec_color = "#5aae78"
    elif overall_pct >= 45:
        recommendation = "Hold / Neutral"
        rec_color = "#d9892b"
    elif overall_pct >= 35:
        recommendation = "Caution"
        rec_color = "#e08c4a"
    else:
        recommendation = "Avoid"
        rec_color = "#c0392b"

    return {
        "segment": segment,
        "listingPrice": listing_price,
        "listingPpsf": round(listing_ppsf, 2),
        "fairValuePpsf": round(fair_value_ppsf, 2),
        "fairValueTotal": round(fair_value_total, 2),
        "premiumDiscountPct": round(premium_discount_pct, 2),
        "overallScore": round(overall_pct, 1),
        "recommendation": recommendation,
        "recommendationColor": rec_color,
        "signals": signals,
        "locality": locality,
        "city": city,
        "metroDistanceKm": round(metro_km, 2) if metro_valid else None,
    }


@app.post("/property-intelligence/analyze")
def property_intelligence_bf(req: PIBfRequest) -> dict[str, Any]:
    """Property intelligence for a Builder Floor listing."""
    if bf_model is None:
        raise HTTPException(status_code=503, detail="Builder Floor model not available")
    vor_feats = compute_voronoi_features(req.lat, req.lon, bf_vor) if bf_vor is not None else {"voronoi_dist_to_seed": 0.0}
    X = build_features(
        bhk=req.bhk, area_sqft=req.area_sqft, bathrooms=req.bathrooms, balconies=req.balconies,
        circle_rate=req.circle_rate, is_parking=req.is_parking, is_pool=req.is_pool,
        is_main_road=req.is_main_road, is_garden_park=req.is_garden_park, is_gated=req.is_gated,
        is_corner=req.is_corner, age=req.age, furnishing=req.furnishing, facing=req.facing,
        voronoi_feats=vor_feats, model=bf_model,
    )
    pred_log = float(bf_model.predict(X)[0])
    pred_ratio = float(np.expm1(pred_log))
    fair_ppsf = float(pred_ratio * req.circle_rate)
    fair_total = float(fair_ppsf * req.area_sqft)
    voronoi_dist = float(vor_feats.get("voronoi_dist_to_seed", 5.0))
    conf = float(np.clip(1.0 - voronoi_dist / 20.0, 0.3, 0.95))
    return _build_property_intelligence(
        segment="builder-floor", listing_price=req.listing_price,
        fair_value_ppsf=fair_ppsf, fair_value_total=fair_total,
        area_sqft=req.area_sqft, circle_rate=req.circle_rate, lat=req.lat, lon=req.lon,
        locality=req.locality, city=req.city,
        bhk=req.bhk, bathrooms=req.bathrooms, balconies=req.balconies, age=req.age,
        is_gated=req.is_gated, is_parking=req.is_parking, is_pool=req.is_pool,
        is_garden_park=req.is_garden_park, is_main_road=req.is_main_road, is_corner=req.is_corner,
        furnishing=req.furnishing, facing=req.facing,
        voronoi_dist=voronoi_dist, model_confidence=conf, ncr_stats=NCR_STATS,
    )


@app.post("/property-intelligence/analyze-apartment")
def property_intelligence_apt(req: PIAptRequest) -> dict[str, Any]:
    """Property intelligence for an Apartment listing."""
    if apt_model is None:
        raise HTTPException(status_code=503, detail="Apartment model not available")
    vor_feats = compute_voronoi_features(req.lat, req.lon, apt_vor) if apt_vor is not None else {"voronoi_dist_to_seed": 0.0}
    floor_feats = {
        "floor_low": 1 if "Low" in req.floor_level else 0,
        "floor_medium": 1 if "Medium" in req.floor_level else 0,
        "floor_high": 1 if "High" in req.floor_level else 0,
        "is_ground_floor": int(req.is_ground),
        "is_top_floor": int(req.is_top),
    }
    floor_feats.update(build_apartment_property_features(req.property_segment, apt_model))
    X = build_features(
        bhk=req.bhk, area_sqft=req.area_sqft, bathrooms=req.bathrooms, balconies=req.balconies,
        circle_rate=req.circle_rate, is_parking=req.is_parking, is_pool=req.is_pool,
        is_main_road=req.is_main_road, is_garden_park=req.is_garden_park, is_gated=req.is_gated,
        is_corner=req.is_corner, age=req.age, furnishing=req.furnishing, facing=req.facing,
        voronoi_feats=vor_feats, model=apt_model, floor_feats=floor_feats,
    )
    pred_log = float(apt_model.predict(X)[0])
    pred_ratio = float(np.expm1(pred_log))
    fair_ppsf = float(pred_ratio * req.circle_rate)
    fair_total = float(fair_ppsf * req.area_sqft)
    voronoi_dist = float(vor_feats.get("voronoi_dist_to_seed", 5.0))
    conf = float(np.clip(1.0 - voronoi_dist / 20.0, 0.3, 0.95))
    return _build_property_intelligence(
        segment="apartment", listing_price=req.listing_price,
        fair_value_ppsf=fair_ppsf, fair_value_total=fair_total,
        area_sqft=req.area_sqft, circle_rate=req.circle_rate, lat=req.lat, lon=req.lon,
        locality=req.locality, city=req.city,
        bhk=req.bhk, bathrooms=req.bathrooms, balconies=req.balconies, age=req.age,
        is_gated=req.is_gated, is_parking=req.is_parking, is_pool=req.is_pool,
        is_garden_park=req.is_garden_park, is_main_road=req.is_main_road, is_corner=req.is_corner,
        furnishing=req.furnishing, facing=req.facing, floor_level=req.floor_level,
        property_segment=req.property_segment,
        voronoi_dist=voronoi_dist, model_confidence=conf, ncr_stats=NCR_STATS,
    )


@app.post("/property-intelligence/analyze-plot")
def property_intelligence_plot(req: PIPlotRequest) -> dict[str, Any]:
    """Property intelligence for a Plot listing."""
    if plot_model is None:
        raise HTTPException(status_code=503, detail="Plot model not available")
    X = build_plot_features(
        area_sqft=req.area_sqft, circle_rate=req.circle_rate, latitude=req.lat, longitude=req.lon,
        usage_type=req.usage_type, facing_direction=req.facing_direction,
        is_park_facing=req.is_park_facing, is_corner=req.is_corner, is_rectangular=req.is_rectangular,
        is_gated=req.is_gated, has_boundary_wall=req.has_boundary_wall,
        road_width_upto_9m=req.road_width_upto_9m, road_width_9_to_18m=req.road_width_9_to_18m,
        road_width_18_plus=req.road_width_18_plus, model_bundle=plot_bundle, road_segments=road_segments,
    )
    pred_log_ppsf = float(plot_model.predict(X)[0])
    fair_ppsf = float(np.expm1(pred_log_ppsf))
    fair_total = float(fair_ppsf * req.area_sqft)
    return _build_property_intelligence(
        segment="plot", listing_price=req.listing_price,
        fair_value_ppsf=fair_ppsf, fair_value_total=fair_total,
        area_sqft=req.area_sqft, circle_rate=req.circle_rate, lat=req.lat, lon=req.lon,
        locality=req.locality, city=req.city,
        is_gated=req.is_gated, is_corner=req.is_corner, is_rectangular=req.is_rectangular,
        has_boundary_wall=req.has_boundary_wall, is_park_facing=req.is_park_facing,
        road_width_18_plus=req.road_width_18_plus, road_width_9_to_18m=req.road_width_9_to_18m,
        usage_type=req.usage_type, model_confidence=0.75, ncr_stats=NCR_STATS,
    )


# ── Layer 2: Property Listing & Analytics ─────────────────────────────────────

_PROP_SEG_CFG: dict[str, dict] = {
    "builder-floor": {
        "dir": "builder_floor",
        "csv": str(PROJECT_ROOT / "inputs" / "builder_floor_with_pi.csv"),
        "has_ids": True,
        "area_col": "covered_area_sqft",
    },
    "apartment": {
        "dir": "apt",
        "csv": str(PROJECT_ROOT / "inputs" / "apartment_with_pi.csv"),
        "has_ids": True,
        "area_col": "covered_area_sqft",
    },
    "plot": {
        "dir": "plot",
        "csv": str(PROJECT_ROOT / "inputs" / "plot_with_pi.csv"),
        "has_ids": False,
        "area_col": "plot_area",
    },
}

# Lazy-loaded cache for the property CSVs (loaded once per segment).
_prop_df_cache: dict[str, pd.DataFrame] = {}
_prop_df_lock = threading.Lock()


def _load_prop_df(segment: str) -> pd.DataFrame:
    if segment not in _prop_df_cache:
        with _prop_df_lock:
            if segment not in _prop_df_cache:
                cfg = _PROP_SEG_CFG[segment]
                _prop_df_cache[segment] = pd.read_csv(cfg["csv"], low_memory=False)
    return _prop_df_cache[segment]


def _load_rho_df(segment: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "opt" / _PROP_SEG_CFG[segment]["dir"] / "rho_details.csv"
    return pd.read_csv(path, low_memory=False)


def _load_forecast_df(segment: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "opt" / _PROP_SEG_CFG[segment]["dir"] / "property_forecasts_flodata.csv"
    return pd.read_csv(path, low_memory=False)


def _validate_segment(segment: str) -> None:
    if segment not in _PROP_SEG_CFG:
        raise HTTPException(status_code=400, detail=f"segment must be one of: {list(_PROP_SEG_CFG)}")


@app.get("/properties/list")
def properties_list(
    segment: str = Query(..., description="builder-floor | apartment | plot"),
    city: str = Query(None, description="Filter by city (case-insensitive partial match)"),
    locality: str = Query(None, description="Filter by locality (fuzzy match)"),
    limit: int = Query(50, ge=1, le=500, description="Max rows to return"),
) -> dict[str, Any]:
    """List properties for a segment, optionally filtered by city/locality."""
    _validate_segment(segment)
    cfg = _PROP_SEG_CFG[segment]
    df = _load_prop_df(segment).copy()

    if city:
        df = df[df["city"].str.lower().str.contains(city.lower(), na=False)]

    if locality:
        if "locality" in df.columns:
            candidates = df["locality"].dropna().unique().tolist()
            best_matches = [c for c in candidates if _locality_match_score(locality, c) >= 0.35]
            if best_matches:
                df = df[df["locality"].isin(best_matches)]

    area_col = cfg["area_col"]
    rows = []
    for _, r in df.head(limit).iterrows():
        row: dict[str, Any] = {
            "locality": r.get("locality"),
            "city":     r.get("city"),
            "area_sqft": r.get(area_col),
            "listed_ppsf": round(float(r["price_per_sqft"]), 0) if pd.notna(r.get("price_per_sqft")) else None,
            "model_ppsf":  round(float(r["pi_price_per_sqft"]), 0) if pd.notna(r.get("pi_price_per_sqft")) else None,
        }
        if cfg["has_ids"]:
            row["property_id"] = r.get("property_id")
            listed = r.get("price_per_sqft")
            model  = r.get("pi_price_per_sqft")
            if pd.notna(listed) and pd.notna(model) and float(model) != 0:
                row["delta_pct"] = round((float(listed) - float(model)) / float(model) * 100, 2)
        if "bhk" in df.columns:
            row["bhk"] = r.get("bhk")
        rows.append(row)

    return {"segment": segment, "count": len(rows), "items": rows}


@app.get("/properties/summary")
def properties_summary(
    segment: str = Query(..., description="builder-floor | apartment"),
    property_id: str = Query(..., description="Property ID from /properties/list"),
) -> dict[str, Any]:
    """Return full property row by property_id (builder-floor / apartment only)."""
    _validate_segment(segment)
    if not _PROP_SEG_CFG[segment]["has_ids"]:
        raise HTTPException(status_code=400, detail="plot segment has no individual property IDs; use /properties/list?segment=plot")

    df = _load_prop_df(segment)
    matches = df[df["property_id"] == property_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"property_id '{property_id}' not found in segment '{segment}'")

    r = matches.iloc[0]
    result: dict[str, Any] = r.dropna().to_dict()
    listed = r.get("price_per_sqft")
    model  = r.get("pi_price_per_sqft")
    if pd.notna(listed) and pd.notna(model) and float(model) != 0:
        result["delta_pct_listed_vs_model"] = round((float(listed) - float(model)) / float(model) * 100, 2)

    return {"segment": segment, "property_id": property_id, "data": result}


@app.get("/properties/rho")
def properties_rho(
    segment: str = Query(..., description="builder-floor | apartment | plot"),
    property_id: str = Query(None, description="Property ID (builder-floor / apartment)"),
    locality: str = Query(None, description="Locality name (plot segment)"),
) -> dict[str, Any]:
    """Return rho blending components for a property or locality."""
    _validate_segment(segment)
    df = _load_rho_df(segment)

    if _PROP_SEG_CFG[segment]["has_ids"]:
        if not property_id:
            raise HTTPException(status_code=400, detail="property_id is required for segment builder-floor / apartment")
        matches = df[df["property_id"] == property_id]
        if matches.empty:
            raise HTTPException(status_code=404, detail=f"property_id '{property_id}' not found in rho data for segment '{segment}'")
        r = matches.iloc[0]
        return {
            "segment": segment,
            "property_id": property_id,
            "locality": r.get("locality"),
            "listed_ppsf": round(float(r["price_per_sqft"]), 0) if pd.notna(r.get("price_per_sqft")) else None,
            "model_ppsf":  round(float(r["pi_price_per_sqft"]), 0) if pd.notna(r.get("pi_price_per_sqft")) else None,
            "comp_support":       round(float(r["comp_support"]), 4) if pd.notna(r.get("comp_support")) else None,
            "uniqueness":         round(float(r["uniqueness"]), 4) if pd.notna(r.get("uniqueness")) else None,
            "model_confidence":   round(float(r["model_confidence"]), 4) if pd.notna(r.get("model_confidence")) else None,
            "rho_0":              round(float(r["rho_0"]), 4) if pd.notna(r.get("rho_0")) else None,
        }
    else:
        # Plot: locality-level rho
        if not locality:
            raise HTTPException(status_code=400, detail="locality is required for plot segment")
        candidates = df["locality"].dropna().unique().tolist()
        best, score = _best_locality_match(locality, candidates)
        if score < 0.35:
            raise HTTPException(status_code=404, detail=f"Locality '{locality}' not found in plot rho data")
        r = df[df["locality"] == best].iloc[0]
        return {
            "segment": segment,
            "locality": best,
            "listed_ppsf": round(float(r["price_per_sqft"]), 0) if pd.notna(r.get("price_per_sqft")) else None,
            "model_ppsf":  round(float(r["pi_price_per_sqft"]), 0) if pd.notna(r.get("pi_price_per_sqft")) else None,
            "comp_support":     round(float(r["comp_support"]), 4) if pd.notna(r.get("comp_support")) else None,
            "uniqueness":       round(float(r["uniqueness"]), 4) if pd.notna(r.get("uniqueness")) else None,
            "model_confidence": round(float(r["model_confidence"]), 4) if pd.notna(r.get("model_confidence")) else None,
            "rho_0":            round(float(r["rho_0"]), 4) if pd.notna(r.get("rho_0")) else None,
        }


@app.get("/properties/yoy")
def properties_yoy(
    segment: str = Query(..., description="builder-floor | apartment | plot"),
    property_id: str = Query(..., description="Property ID from forecast data"),
    years: int = Query(5, ge=1, le=10, description="Number of forecast years"),
) -> dict[str, Any]:
    """Return annual YoY forecast price changes for a specific property."""
    _validate_segment(segment)
    df = _load_forecast_df(segment)
    prop_df = df[df["property_id"] == property_id]
    if prop_df.empty:
        raise HTTPException(status_code=404, detail=f"property_id '{property_id}' not found in forecast data for segment '{segment}'")

    qmap: dict[int, float] = dict(zip(prop_df["quarter"].astype(int), prop_df["forecast_price_per_sqft"].astype(float)))
    dmap: dict[int, str]   = dict(zip(prop_df["quarter"].astype(int), prop_df["date"].astype(str)))

    yoy_results = []
    for n in range(1, years + 1):
        end_q   = 4 * n
        start_q = max(1, 4 * (n - 1))  # year 1: Q1 as base (earliest forecast quarter)
        if end_q in qmap and start_q in qmap:
            sp = float(qmap[start_q])
            ep = float(qmap[end_q])
            yoy_pct = round((ep - sp) / sp * 100, 2) if sp != 0 else None
            yoy_results.append({
                "label":        f"Y+{n}",
                "start_date":   dmap.get(start_q),
                "end_date":     dmap.get(end_q),
                "anchor_ppsf":  round(sp),
                "target_ppsf":  round(ep),
                "yoy_pct":      yoy_pct,
            })

    locality = prop_df["locality"].iloc[0] if "locality" in prop_df.columns else None
    return {
        "segment":     segment,
        "property_id": property_id,
        "locality":    locality,
        "yoy":         yoy_results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
