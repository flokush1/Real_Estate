import sys
from real_estate.components.data_ingestion import DataIngestion
from real_estate.components.data_transformation import DataTransformation
from real_estate.constant import RENT_DATA_FILES
from real_estate.entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    RentDataIngestionConfig,
    RentDataTransformationConfig,
)
from real_estate.exception.exception import RealEstateException
from real_estate.logging.logger import logging


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
