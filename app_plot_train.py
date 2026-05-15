from real_estate.logging.logger import logging
from real_estate.pipeline.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    logging.info("Plot training pipeline application started")
    pipeline = TrainingPipeline()
    pipeline.run_plot_training_pipeline()
    logging.info("Plot training pipeline application finished")
