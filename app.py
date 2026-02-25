from real_estate.pipeline.training_pipeline import TrainingPipeline
from real_estate.logging.logger import logging

if __name__ == "__main__":
    logging.info("Application started")
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
    logging.info("Application finished")
