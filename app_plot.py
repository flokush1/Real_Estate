from real_estate.logging.logger import logging
from real_estate.pipeline.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    logging.info("Plot pipeline application started")
    pipeline = TrainingPipeline()
    pipeline.run_plot_pipeline()
    logging.info("Plot pipeline application finished")
