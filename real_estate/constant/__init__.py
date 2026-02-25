import os

# ─── Data paths ──────────────────────────────────────────────
DATA_DIR = "real_estate_data"
RAW_DATA_FILES = [
    "2026-02-23_magicbricks_new.csv",
    "2026-02-25_housing_new.csv",
    "2026-02-25_12-29-27_magicbricks_new.csv",
]
CITY_LOCALITIES_JSON = os.path.join(DATA_DIR, "ncr_colonies.json")

# ─── Artifact paths ─────────────────────────────────────────
ARTIFACT_DIR = "artifact"
DATA_INGESTION_DIR = "data_ingestion"
DATA_INGESTION_RAW_DIR = "raw"
DATA_INGESTION_MERGED_DIR = "merged"
MERGED_FILE_NAME = "merged.csv"

DATA_TRANSFORMATION_DIR = "data_transformation"
TRANSFORMED_FILE_NAME = "transformed.csv"

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
