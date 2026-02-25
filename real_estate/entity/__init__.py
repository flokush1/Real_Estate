import os
from dataclasses import dataclass
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
class ModelTrainerArtifact:
    model_file_path: str
