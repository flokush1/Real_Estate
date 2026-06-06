import os
import shutil
import sys

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False

from real_estate.components.data_ingestion import DataIngestion
from real_estate.components.data_transformation import DataTransformation
from real_estate.components.plot_data_ingestion import PlotDataIngestion
from real_estate.components.plot_data_transformation import PlotDataTransformation
from real_estate.components.plot_model_trainer import PlotModelTrainer
from real_estate.components.apt_model_trainer import AptModelTrainer
from real_estate.components.bf_model_trainer import BfModelTrainer
from real_estate.components.rent_model_trainer import AptRentModelTrainer, BfRentModelTrainer
from real_estate.constant import RENT_DATA_FILES
from real_estate.entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    PlotDataIngestionConfig,
    PlotModelTrainerConfig,
    PlotDataTransformationConfig,
    RentDataIngestionConfig,
    RentDataTransformationConfig,
    AptModelTrainerConfig,
    BfModelTrainerConfig,
    AptRentModelTrainerConfig,
    BfRentModelTrainerConfig,
)
from real_estate.exception.exception import RealEstateException
from real_estate.logging.logger import logging
from real_estate.utils import get_next_model_version, register_model_version
from real_estate.utils.s3_uploader import upload_artifact_pkls


class TrainingPipeline:
    """Orchestrates: Data Ingestion → Data Transformation → (Model Training)."""

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()

    def start_data_ingestion(self):
        logging.info(">>> Starting Data Ingestion stage")
        data_ingestion = DataIngestion(config=self.data_ingestion_config)
        return data_ingestion.initiate_data_ingestion()

    def start_data_transformation(self, data_ingestion_artifact):
        logging.info(">>> Starting Data Transformation stage")
        data_transformation = DataTransformation(
            data_ingestion_artifact=data_ingestion_artifact,
            config=self.data_transformation_config,
        )
        return data_transformation.initiate_data_transformation()

    def run_pipeline(self):
        try:
            logging.info("============ Training Pipeline Started ============")

            # Stage 1 – Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info(f"Data ingestion complete: {data_ingestion_artifact}")

            # Stage 2 – Data Transformation
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact)
            logging.info(f"Data transformation complete: {data_transformation_artifact}")

            # Stage 3 – Model Training (placeholder for next iteration)
            # model_trainer_artifact = self.start_model_training(data_transformation_artifact)

            logging.info("============ Training Pipeline Finished ============")

        except Exception as e:
            raise RealEstateException(e, sys)

    def run_rent_pipeline(self):
        """Ingest + clean mb_rent.csv / ho_rent.csv and save to artifact/rent_*."""
        try:
            logging.info("============ Rent Pipeline Started ============")

            # Stage 1 – Rent Data Ingestion
            rent_ingestion_config = RentDataIngestionConfig()
            rent_ingestion = DataIngestion(
                config=rent_ingestion_config,
                raw_data_files=RENT_DATA_FILES,
            )
            rent_ingestion_artifact = rent_ingestion.initiate_data_ingestion()
            logging.info(f"Rent data ingestion complete: {rent_ingestion_artifact}")

            # Stage 2 – Rent Data Transformation
            rent_transformation_config = RentDataTransformationConfig()
            rent_transformation = DataTransformation(
                data_ingestion_artifact=rent_ingestion_artifact,
                config=rent_transformation_config,
            )
            rent_transformation_artifact = rent_transformation.initiate_data_transformation()
            logging.info(f"Rent data transformation complete: {rent_transformation_artifact}")

            logging.info("============ Rent Pipeline Finished ============")
            return rent_transformation_artifact

        except Exception as e:
            raise RealEstateException(e, sys)

    def run_plot_pipeline(self):
        """
        Plot/Land pipeline:
        1) Fetch ho_raw_data + mb_raw_data from PostgreSQL
        2) Align and merge plot records
        3) Run plot-specific cleaning and feature engineering
        """
        try:
            logging.info("============ Plot Pipeline Started ============")

            plot_ingestion_config = PlotDataIngestionConfig()
            plot_ingestion = PlotDataIngestion(config=plot_ingestion_config)
            plot_ingestion_artifact = plot_ingestion.initiate_data_ingestion()
            logging.info(f"Plot ingestion complete: {plot_ingestion_artifact}")

            plot_transformation_config = PlotDataTransformationConfig()
            plot_transformation = PlotDataTransformation(
                data_ingestion_artifact=plot_ingestion_artifact,
                config=plot_transformation_config,
            )
            plot_transformation_artifact = plot_transformation.initiate_data_transformation()
            logging.info(f"Plot transformation complete: {plot_transformation_artifact}")

            logging.info("============ Plot Pipeline Finished ============")
            return plot_transformation_artifact

        except Exception as e:
            raise RealEstateException(e, sys)

    def run_plot_training_pipeline(self):
        """
        Plot/Land full training pipeline (versioned):
        1) Fetch fresh data from DB (always — no cache shortcut)
        2) Clean and transform to model-ready features
        3) Train model and save to artifact/plot_model_trainer/v{N}/
        4) Register version in artifact/model_registry.json
        5) Promote artifacts to canonical path so the API stays unchanged
        """
        try:
            logging.info("============ Plot Training Pipeline Started ============")

            # Configure MLflow tracking (SQLite backend, local)
            if _HAS_MLFLOW:
                mlflow.set_tracking_uri("sqlite:///artifact/mlflow.db")
                mlflow.set_experiment("plot-price-model")
                logging.info("MLflow tracking URI: sqlite:///artifact/mlflow.db")

            # Stage 1 – always ingest fresh data
            plot_ingestion_config = PlotDataIngestionConfig()
            plot_ingestion = PlotDataIngestion(config=plot_ingestion_config)
            plot_ingestion_artifact = plot_ingestion.initiate_data_ingestion()
            logging.info(f"Plot ingestion complete: {plot_ingestion_artifact}")

            # Stage 2 – clean and engineer features
            plot_transformation_config = PlotDataTransformationConfig()
            plot_transformation = PlotDataTransformation(
                data_ingestion_artifact=plot_ingestion_artifact,
                config=plot_transformation_config,
            )
            plot_transformation_artifact = plot_transformation.initiate_data_transformation()
            logging.info(f"Plot transformation complete: {plot_transformation_artifact}")

            # Stage 3 – train into a versioned directory
            next_version = get_next_model_version("plot")
            logging.info(f"Training plot model version: v{next_version}")
            versioned_config = PlotModelTrainerConfig(version=next_version)
            os.makedirs(versioned_config.model_dir, exist_ok=True)

            plot_model_trainer = PlotModelTrainer(
                data_transformation_artifact=plot_transformation_artifact,
                config=versioned_config,
            )
            plot_model_trainer_artifact = plot_model_trainer.initiate_model_training()
            logging.info(f"Plot model training complete: {plot_model_trainer_artifact}")

            # Stage 4 – record in model registry
            register_model_version(
                model_type="plot",
                version=next_version,
                model_path=versioned_config.model_file_path,
                metrics={
                    "mae": plot_model_trainer_artifact.mae,
                    "mape": plot_model_trainer_artifact.mape,
                    "r2": plot_model_trainer_artifact.r2,
                    "best_model": plot_model_trainer_artifact.best_model_name,
                },
            )

            # Stage 5 – promote to canonical path so the API always loads latest
            canonical_config = PlotModelTrainerConfig(version=0)
            os.makedirs(canonical_config.model_dir, exist_ok=True)
            shutil.copy2(versioned_config.model_file_path, canonical_config.model_file_path)
            shutil.copy2(
                versioned_config.feature_columns_file_path,
                canonical_config.feature_columns_file_path,
            )
            logging.info(
                f"Promoted v{next_version} → canonical: {canonical_config.model_file_path}"
            )

            logging.info("============ Plot Training Pipeline Finished ============")
            upload_artifact_pkls(categories=["plot"])
            return plot_model_trainer_artifact

        except Exception as e:
            raise RealEstateException(e, sys)

    # ════════════════════════════════════════════════════════
    #  APARTMENT TRAINING PIPELINE
    # ════════════════════════════════════════════════════════
    def run_apt_training_pipeline(self):
        """
        Apartment full training pipeline (versioned):
        1) Ingest raw data from CSV sources (ho/mb)
        2) Clean + feature-engineer all property types
        3) Train apartment model (filters cleaned data)
        4) Save to artifact/apt_model_trainer/v{N}/
        5) Register in model_registry.json
        6) Promote to canonical path
        """
        try:
            logging.info("============ Apt Training Pipeline Started ============")

            if _HAS_MLFLOW:
                mlflow.set_tracking_uri("sqlite:///artifact/mlflow.db")
                mlflow.set_experiment("apartment-price-model")

            next_version = get_next_model_version("apt")
            logging.info(f"Training apartment model version: v{next_version}")

            # Stage 1 – ingest fresh raw data
            ingestion_config = DataIngestionConfig(version=next_version)
            ingestion = DataIngestion(config=ingestion_config)
            ingestion_artifact = ingestion.initiate_data_ingestion()
            logging.info(f"Apt ingestion complete: {ingestion_artifact}")

            # Stage 2 – clean + feature engineering
            transformation_config = DataTransformationConfig(version=next_version)
            transformation = DataTransformation(
                data_ingestion_artifact=ingestion_artifact,
                config=transformation_config,
            )
            transformation_artifact = transformation.initiate_data_transformation()
            logging.info(f"Apt transformation complete: {transformation_artifact}")

            # Stage 3 – train versioned model
            apt_config = AptModelTrainerConfig(version=next_version)
            os.makedirs(apt_config.model_dir, exist_ok=True)
            trainer = AptModelTrainer(
                data_transformation_artifact=transformation_artifact,
                config=apt_config,
            )
            artifact = trainer.initiate_model_training()
            logging.info(f"Apt training complete: {artifact}")

            # Stage 4 – register
            register_model_version(
                model_type="apt",
                version=next_version,
                model_path=apt_config.model_file_path,
                metrics={
                    "mae": artifact.mae,
                    "mape": artifact.mape,
                    "r2": artifact.r2,
                    "best_model": artifact.best_model_name,
                    "data_version": next_version,
                    "cleaned_csv": transformation_config.transformed_file_path,
                    "merged_csv": ingestion_config.merged_file_path,
                },
            )

            # Stage 5 – promote to canonical path (API loads these)
            canonical = AptModelTrainerConfig(version=0)
            os.makedirs(canonical.model_dir, exist_ok=True)
            shutil.copy2(apt_config.model_file_path, canonical.model_file_path)
            shutil.copy2(apt_config.feature_columns_file_path, canonical.feature_columns_file_path)
            shutil.copy2(apt_config.voronoi_file_path, canonical.voronoi_file_path)
            logging.info(f"Promoted apt v{next_version} → canonical: {canonical.model_file_path}")

            logging.info("============ Apt Training Pipeline Finished ============")
            upload_artifact_pkls(categories=["apt"], subcategories=["sell"])
            return artifact

        except Exception as e:
            raise RealEstateException(e, sys)

    # ════════════════════════════════════════════════════════
    #  BUILDER FLOOR TRAINING PIPELINE
    # ════════════════════════════════════════════════════════
    def run_bf_training_pipeline(self):
        """
        Builder Floor full training pipeline (versioned).
        Same stages as apartment pipeline — filters for builder_floor rows.
        """
        try:
            logging.info("============ BF Training Pipeline Started ============")

            if _HAS_MLFLOW:
                mlflow.set_tracking_uri("sqlite:///artifact/mlflow.db")
                mlflow.set_experiment("builder-floor-price-model")

            next_version = get_next_model_version("bf")
            logging.info(f"Training builder floor model version: v{next_version}")

            # Stage 1
            ingestion_config = DataIngestionConfig(version=next_version)
            ingestion = DataIngestion(config=ingestion_config)
            ingestion_artifact = ingestion.initiate_data_ingestion()
            logging.info(f"BF ingestion complete: {ingestion_artifact}")

            # Stage 2
            transformation_config = DataTransformationConfig(version=next_version)
            transformation = DataTransformation(
                data_ingestion_artifact=ingestion_artifact,
                config=transformation_config,
            )
            transformation_artifact = transformation.initiate_data_transformation()
            logging.info(f"BF transformation complete: {transformation_artifact}")

            # Stage 3
            bf_config = BfModelTrainerConfig(version=next_version)
            os.makedirs(bf_config.model_dir, exist_ok=True)
            trainer = BfModelTrainer(
                data_transformation_artifact=transformation_artifact,
                config=bf_config,
            )
            artifact = trainer.initiate_model_training()
            logging.info(f"BF training complete: {artifact}")

            # Stage 4
            register_model_version(
                model_type="bf",
                version=next_version,
                model_path=bf_config.model_file_path,
                metrics={
                    "mae": artifact.mae,
                    "mape": artifact.mape,
                    "r2": artifact.r2,
                    "best_model": artifact.best_model_name,
                    "data_version": next_version,
                    "cleaned_csv": transformation_config.transformed_file_path,
                    "merged_csv": ingestion_config.merged_file_path,
                },
            )

            # Stage 5
            canonical = BfModelTrainerConfig(version=0)
            os.makedirs(canonical.model_dir, exist_ok=True)
            shutil.copy2(bf_config.model_file_path, canonical.model_file_path)
            shutil.copy2(bf_config.feature_columns_file_path, canonical.feature_columns_file_path)
            shutil.copy2(bf_config.voronoi_file_path, canonical.voronoi_file_path)
            logging.info(f"Promoted bf v{next_version} → canonical: {canonical.model_file_path}")

            logging.info("============ BF Training Pipeline Finished ============")
            upload_artifact_pkls(categories=["bf"], subcategories=["sell"])
            return artifact

        except Exception as e:
            raise RealEstateException(e, sys)

    # ════════════════════════════════════════════════════════
    #  APARTMENT RENT TRAINING PIPELINE
    # ════════════════════════════════════════════════════════
    def run_apt_rent_training_pipeline(self):
        """
        Apartment Rent training pipeline:
        1) Ingest + transform rent data (ho_rent + mb_rent)
        2) Train apartment rent model (log1p monthly rent target)
        3) Save to artifact/apt_rent_model_trainer/
        """
        try:
            logging.info("============ Apt Rent Training Pipeline Started ============")

            # Stage 1 – Rent data ingestion + transformation
            rent_artifact = self.run_rent_pipeline()
            logging.info(f"Rent data ready: {rent_artifact.transformed_file_path}")

            # Stage 2 – Train
            apt_rent_config = AptRentModelTrainerConfig(version=0)
            os.makedirs(apt_rent_config.model_dir, exist_ok=True)
            trainer = AptRentModelTrainer(
                data_transformation_artifact=rent_artifact,
                config=apt_rent_config,
            )
            artifact = trainer.initiate_model_training()
            logging.info(f"Apt rent training complete: {artifact}")

            logging.info("============ Apt Rent Training Pipeline Finished ============")
            upload_artifact_pkls(categories=["apt"], subcategories=["rent"])
            return artifact

        except Exception as e:
            raise RealEstateException(e, sys)

    # ════════════════════════════════════════════════════════
    #  BUILDER FLOOR RENT TRAINING PIPELINE
    # ════════════════════════════════════════════════════════
    def run_bf_rent_training_pipeline(self):
        """
        Builder Floor Rent training pipeline:
        1) Ingest + transform rent data (ho_rent + mb_rent)
        2) Train builder floor rent model (log1p monthly rent target)
        3) Save to artifact/bf_rent_model_trainer/
        """
        try:
            logging.info("============ BF Rent Training Pipeline Started ============")

            # Stage 1 – Rent data ingestion + transformation
            rent_artifact = self.run_rent_pipeline()
            logging.info(f"Rent data ready: {rent_artifact.transformed_file_path}")

            # Stage 2 – Train
            bf_rent_config = BfRentModelTrainerConfig(version=0)
            os.makedirs(bf_rent_config.model_dir, exist_ok=True)
            trainer = BfRentModelTrainer(
                data_transformation_artifact=rent_artifact,
                config=bf_rent_config,
            )
            artifact = trainer.initiate_model_training()
            logging.info(f"BF rent training complete: {artifact}")

            logging.info("============ BF Rent Training Pipeline Finished ============")
            upload_artifact_pkls(categories=["bf"], subcategories=["rent"])
            return artifact

        except Exception as e:
            raise RealEstateException(e, sys)
