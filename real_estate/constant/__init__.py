import os


def _resolve_data_dir() -> str:
    """Resolve real_estate_data regardless of the current working directory."""
    cwd_candidate = os.path.abspath(os.path.join(os.getcwd(), "real_estate_data"))
    if os.path.isdir(cwd_candidate):
        return cwd_candidate

    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "real_estate_data")
    )

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
