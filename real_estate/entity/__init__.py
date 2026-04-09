import os
from dataclasses import dataclass, field
from real_estate.constant import (
    ARTIFACT_DIR,
    DATA_INGESTION_DIR,
    DATA_INGESTION_RAW_DIR,
    DATA_INGESTION_MERGED_DIR,
    MERGED_FILE_NAME,
    DATA_TRANSFORMATION_DIR,
    TRANSFORMED_FILE_NAME,
    MODEL_TRAINER_DIR,
    MODEL_FILE_NAME,
    RENT_INGESTION_DIR,
    RENT_MERGED_FILE_NAME,
    RENT_TRANSFORMATION_DIR,
    RENT_TRANSFORMED_FILE_NAME,
)


# ═══════════════════════════════════════════════════════════════
# CONFIG – what settings do we *want* for each stage
# ═══════════════════════════════════════════════════════════════

@dataclass
class DataIngestionConfig:
    raw_data_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR, DATA_INGESTION_RAW_DIR)
    merged_data_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR, DATA_INGESTION_MERGED_DIR)
    merged_file_path: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR, DATA_INGESTION_MERGED_DIR, MERGED_FILE_NAME)


@dataclass
class DataTransformationConfig:
    transformed_data_dir: str = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR)
    transformed_file_path: str = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR, TRANSFORMED_FILE_NAME)
    price_per_sqft_min: float = 800.0


@dataclass
class RentDataIngestionConfig:
    raw_data_dir: str = os.path.join(ARTIFACT_DIR, RENT_INGESTION_DIR, DATA_INGESTION_RAW_DIR)
    merged_data_dir: str = os.path.join(ARTIFACT_DIR, RENT_INGESTION_DIR, DATA_INGESTION_MERGED_DIR)
    merged_file_path: str = os.path.join(ARTIFACT_DIR, RENT_INGESTION_DIR, DATA_INGESTION_MERGED_DIR, RENT_MERGED_FILE_NAME)


@dataclass
class RentDataTransformationConfig:
    transformed_data_dir: str = os.path.join(ARTIFACT_DIR, RENT_TRANSFORMATION_DIR)
    transformed_file_path: str = os.path.join(ARTIFACT_DIR, RENT_TRANSFORMATION_DIR, RENT_TRANSFORMED_FILE_NAME)
    price_per_sqft_min: float = 5.0  # monthly rent / sqft is much lower than sale price / sqft


@dataclass
class ModelTrainerConfig:
    model_dir: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR)
    model_file_path: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR, MODEL_FILE_NAME)


# ═══════════════════════════════════════════════════════════════
# ARTIFACT – what each stage *produces* (paths to outputs)
# ═══════════════════════════════════════════════════════════════

@dataclass
class DataIngestionArtifact:
    merged_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_file_path: str


@dataclass
class RentDataIngestionArtifact:
    merged_file_path: str


@dataclass
class RentDataTransformationArtifact:
    transformed_file_path: str


@dataclass
class ModelTrainerArtifact:
    model_file_path: str
