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
    PLOT_INGESTION_DIR,
    PLOT_INGESTION_RAW_DIR,
    PLOT_INGESTION_MERGED_DIR,
    PLOT_MERGED_FILE_NAME,
    PLOT_TRANSFORMATION_DIR,
    PLOT_TRANSFORMED_FILE_NAME,
    PLOT_MODEL_TRAINER_DIR,
    PLOT_MODEL_FILE_NAME,
    PLOT_FEATURE_COLUMNS_FILE_NAME,
    PLOT_ACTUAL_PREDICTION_HTML,
    PLOT_RESIDUAL_HTML,
    APT_MODEL_TRAINER_DIR,
    APT_MODEL_FILE_NAME,
    APT_FEATURE_COLUMNS_FILE_NAME,
    APT_VORONOI_FILE_NAME,
    BF_MODEL_TRAINER_DIR,
    BF_MODEL_FILE_NAME,
    BF_FEATURE_COLUMNS_FILE_NAME,
    BF_VORONOI_FILE_NAME,
)


# ═══════════════════════════════════════════════════════════════
# CONFIG – what settings do we *want* for each stage
# ═══════════════════════════════════════════════════════════════

@dataclass
class DataIngestionConfig:
    # version=0 → canonical path; version>0 → artifact/.../v{N}/ subfolder
    version: int = 0
    raw_data_dir: str = field(default="", init=False)
    merged_data_dir: str = field(default="", init=False)
    merged_file_path: str = field(default="", init=False)

    def __post_init__(self):
        vdir = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR, f"v{self.version}") if self.version > 0 else os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR)
        self.raw_data_dir = os.path.join(vdir, DATA_INGESTION_RAW_DIR)
        self.merged_data_dir = os.path.join(vdir, DATA_INGESTION_MERGED_DIR)
        self.merged_file_path = os.path.join(vdir, DATA_INGESTION_MERGED_DIR, MERGED_FILE_NAME)


@dataclass
class DataTransformationConfig:
    # version=0 → canonical path; version>0 → artifact/.../v{N}/
    version: int = 0
    price_per_sqft_min: float = 800.0
    transformed_data_dir: str = field(default="", init=False)
    transformed_file_path: str = field(default="", init=False)

    def __post_init__(self):
        vdir = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR, f"v{self.version}") if self.version > 0 else os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR)
        self.transformed_data_dir = vdir
        self.transformed_file_path = os.path.join(vdir, TRANSFORMED_FILE_NAME)


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
class PlotDataIngestionConfig:
    version: int = 0
    raw_data_dir: str = field(default="", init=False)
    merged_data_dir: str = field(default="", init=False)
    merged_file_path: str = field(default="", init=False)

    def __post_init__(self):
        vdir = os.path.join(ARTIFACT_DIR, PLOT_INGESTION_DIR, f"v{self.version}") if self.version > 0 else os.path.join(ARTIFACT_DIR, PLOT_INGESTION_DIR)
        self.raw_data_dir = os.path.join(vdir, PLOT_INGESTION_RAW_DIR)
        self.merged_data_dir = os.path.join(vdir, PLOT_INGESTION_MERGED_DIR)
        self.merged_file_path = os.path.join(vdir, PLOT_INGESTION_MERGED_DIR, PLOT_MERGED_FILE_NAME)


@dataclass
class PlotDataTransformationConfig:
    version: int = 0
    transformed_data_dir: str = field(default="", init=False)
    transformed_file_path: str = field(default="", init=False)

    def __post_init__(self):
        vdir = os.path.join(ARTIFACT_DIR, PLOT_TRANSFORMATION_DIR, f"v{self.version}") if self.version > 0 else os.path.join(ARTIFACT_DIR, PLOT_TRANSFORMATION_DIR)
        self.transformed_data_dir = vdir
        self.transformed_file_path = os.path.join(vdir, PLOT_TRANSFORMED_FILE_NAME)


@dataclass
class PlotModelTrainerConfig:
    # version=0 → canonical (unversioned) path; version>0 → artifact/.../v{N}/ subfolder
    version: int = 0
    random_state: int = 42
    n_clusters: int = 60
    contamination: float = 0.02
    test_size: float = 0.2
    # Paths are computed in __post_init__ based on version
    model_dir: str = field(default="", init=False)
    model_file_path: str = field(default="", init=False)
    feature_columns_file_path: str = field(default="", init=False)
    actual_vs_predicted_file_path: str = field(default="", init=False)
    residual_analysis_file_path: str = field(default="", init=False)

    def __post_init__(self):
        if self.version > 0:
            version_dir = os.path.join(ARTIFACT_DIR, PLOT_MODEL_TRAINER_DIR, f"v{self.version}")
        else:
            version_dir = os.path.join(ARTIFACT_DIR, PLOT_MODEL_TRAINER_DIR)
        self.model_dir = version_dir
        self.model_file_path = os.path.join(version_dir, PLOT_MODEL_FILE_NAME)
        self.feature_columns_file_path = os.path.join(version_dir, PLOT_FEATURE_COLUMNS_FILE_NAME)
        self.actual_vs_predicted_file_path = os.path.join(version_dir, PLOT_ACTUAL_PREDICTION_HTML)
        self.residual_analysis_file_path = os.path.join(version_dir, PLOT_RESIDUAL_HTML)


@dataclass
class ModelTrainerConfig:
    model_dir: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR)
    model_file_path: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR, MODEL_FILE_NAME)


@dataclass
class AptModelTrainerConfig:
    # version=0 → canonical path; version>0 → artifact/apt_model_trainer/v{N}/
    version: int = 0
    random_state: int = 42
    n_voronoi_clusters: int = 60
    contamination: float = 0.02
    test_size: float = 0.2
    n_locality_tiers: int = 4
    model_dir: str = field(default="", init=False)
    model_file_path: str = field(default="", init=False)
    feature_columns_file_path: str = field(default="", init=False)
    voronoi_file_path: str = field(default="", init=False)

    def __post_init__(self):
        vdir = os.path.join(ARTIFACT_DIR, APT_MODEL_TRAINER_DIR, f"v{self.version}") if self.version > 0 else os.path.join(ARTIFACT_DIR, APT_MODEL_TRAINER_DIR)
        self.model_dir = vdir
        self.model_file_path = os.path.join(vdir, APT_MODEL_FILE_NAME)
        self.feature_columns_file_path = os.path.join(vdir, APT_FEATURE_COLUMNS_FILE_NAME)
        self.voronoi_file_path = os.path.join(vdir, APT_VORONOI_FILE_NAME)


@dataclass
class BfModelTrainerConfig:
    # version=0 → canonical path; version>0 → artifact/bf_model_trainer/v{N}/
    version: int = 0
    random_state: int = 42
    n_voronoi_clusters: int = 60
    contamination: float = 0.02
    test_size: float = 0.2
    model_dir: str = field(default="", init=False)
    model_file_path: str = field(default="", init=False)
    feature_columns_file_path: str = field(default="", init=False)
    voronoi_file_path: str = field(default="", init=False)

    def __post_init__(self):
        vdir = os.path.join(ARTIFACT_DIR, BF_MODEL_TRAINER_DIR, f"v{self.version}") if self.version > 0 else os.path.join(ARTIFACT_DIR, BF_MODEL_TRAINER_DIR)
        self.model_dir = vdir
        self.model_file_path = os.path.join(vdir, BF_MODEL_FILE_NAME)
        self.feature_columns_file_path = os.path.join(vdir, BF_FEATURE_COLUMNS_FILE_NAME)
        self.voronoi_file_path = os.path.join(vdir, BF_VORONOI_FILE_NAME)


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
class PlotDataIngestionArtifact:
    merged_file_path: str


@dataclass
class PlotDataTransformationArtifact:
    transformed_file_path: str


@dataclass
class PlotModelTrainerArtifact:
    model_file_path: str
    feature_columns_file_path: str
    best_model_name: str
    mae: float
    mape: float
    r2: float


@dataclass
class ModelTrainerArtifact:
    model_file_path: str


@dataclass
class AptModelTrainerArtifact:
    model_file_path: str
    feature_columns_file_path: str
    voronoi_file_path: str
    best_model_name: str
    mae: float
    mape: float
    r2: float


@dataclass
class BfModelTrainerArtifact:
    model_file_path: str
    feature_columns_file_path: str
    voronoi_file_path: str
    best_model_name: str
    mae: float
    mape: float
    r2: float
