from real_estate.logging.logger import logging
from real_estate.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    logging.info("Apartment training pipeline started")
    pipeline = TrainingPipeline()
    pipeline.run_apt_training_pipeline()
    logging.info("Apartment training pipeline finished")
