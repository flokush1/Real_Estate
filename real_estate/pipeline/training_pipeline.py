import sys
from real_estate.components.data_ingestion import DataIngestion
from real_estate.components.data_transformation import DataTransformation
from real_estate.entity import DataIngestionConfig, DataTransformationConfig
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
