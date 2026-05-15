import os


def _resolve_data_dir() -> str:
    """Resolve real_estate_data regardless of the current working directory."""
    cwd_candidate = os.path.abspath(os.path.join(os.getcwd(), "real_estate_data"))
    if os.path.isdir(cwd_candidate):
        # Check if actual data files are in a nested real_estate_data sub-folder
        nested = os.path.join(cwd_candidate, "real_estate_data")
        if os.path.isdir(nested):
            return nested
        return cwd_candidate

    base = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "real_estate_data")
    )
    nested = os.path.join(base, "real_estate_data")
    if os.path.isdir(nested):
        return nested
    return base

# ─── Data paths ──────────────────────────────────────────────
DATA_DIR = _resolve_data_dir()
RAW_DATA_FILES = [
    "ho_raw_data.csv",
    "mb_raw_data.csv",
]
RENT_DATA_FILES = [
    "ho_rent.csv",
    "mb_rent.csv",
]
CITY_LOCALITIES_JSON = os.path.join(DATA_DIR, "ncr_colonies.json")
CIRCLE_RATES_DIR = os.path.join(DATA_DIR, "circle_rates")

# ─── Artifact paths ─────────────────────────────────────────
ARTIFACT_DIR = "artifact"
DATA_INGESTION_DIR = "data_ingestion"
DATA_INGESTION_RAW_DIR = "raw"
DATA_INGESTION_MERGED_DIR = "merged"
MERGED_FILE_NAME = "merged.csv"

DATA_TRANSFORMATION_DIR = "data_transformation"
TRANSFORMED_FILE_NAME = "cleaned.csv"

RENT_INGESTION_DIR = "rent_ingestion"
RENT_MERGED_FILE_NAME = "rent_merged.csv"
RENT_TRANSFORMATION_DIR = "rent_transformation"
RENT_TRANSFORMED_FILE_NAME = "rent_cleaned.csv"

PLOT_INGESTION_DIR = "plot_ingestion"
PLOT_INGESTION_RAW_DIR = "raw"
PLOT_INGESTION_MERGED_DIR = "merged"
PLOT_MERGED_FILE_NAME = "combined_plot.csv"

PLOT_TRANSFORMATION_DIR = "plot_transformation"
PLOT_TRANSFORMED_FILE_NAME = "cleaning_plot_datav1.csv"
NCR_ROADS_GEOJSON_PATH = os.path.join(DATA_DIR, "NCR Roads.geojson")

PLOT_MODEL_TRAINER_DIR = "plot_model_trainer"
PLOT_MODEL_FILE_NAME = "plot_v3_production_model.pkl"
PLOT_FEATURE_COLUMNS_FILE_NAME = "plot_feature_columns.pkl"
PLOT_ACTUAL_PREDICTION_HTML = "plot_actual_vs_predicted_total_price.html"
PLOT_RESIDUAL_HTML = "plot_residual_analysis.html"

# ─── Apartment model ────────────────────────────────────────
APT_MODEL_TRAINER_DIR = "apt_model_trainer"
APT_MODEL_FILE_NAME = "best_apt_random_forest.pkl"
APT_FEATURE_COLUMNS_FILE_NAME = "apt_feature_columns.pkl"
APT_VORONOI_FILE_NAME = "apt_vor_kmeans.pkl"

# ─── Builder Floor model ────────────────────────────────────
BF_MODEL_TRAINER_DIR = "bf_model_trainer"
BF_MODEL_FILE_NAME = "best_bf_random_forest.pkl"
BF_FEATURE_COLUMNS_FILE_NAME = "bf_feature_columns.pkl"
BF_VORONOI_FILE_NAME = "bf_vor_kmeans.pkl"

MODEL_TRAINER_DIR = "model_trainer"
MODEL_FILE_NAME = "model.pkl"

# ─── MongoDB (optional, for future use) ─────────────────────
MONGODB_URL_KEY = "MONGODB_URL"
DATABASE_NAME = "real_estate_db"
COLLECTION_NAME = "housing_data"

# ─── Training constants ─────────────────────────────────────
TARGET_COLUMN = "price_numeric"
TEST_SIZE = 0.2
RANDOM_STATE = 42
