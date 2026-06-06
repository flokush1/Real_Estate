from dotenv import load_dotenv

load_dotenv()

from real_estate.logging.logger import logging
from real_estate.pipeline.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    logging.info("Master training pipeline started")
    pipeline = TrainingPipeline()

    logging.info("Step 1/5: Apartment sell training")
    pipeline.run_apt_training_pipeline()

    logging.info("Step 2/5: Apartment rent training")
    apt_rent_artifact = pipeline.run_apt_rent_training_pipeline()
    logging.info(f"Apartment rent model saved to: {apt_rent_artifact.model_file_path}")

    logging.info("Step 3/5: Builder floor sell training")
    pipeline.run_bf_training_pipeline()

    logging.info("Step 4/5: Builder floor rent training")
    bf_rent_artifact = pipeline.run_bf_rent_training_pipeline()
    logging.info(f"Builder floor rent model saved to: {bf_rent_artifact.model_file_path}")

    logging.info("Step 5/5: Plot training")
    pipeline.run_plot_training_pipeline()

    logging.info("Master training pipeline finished")
