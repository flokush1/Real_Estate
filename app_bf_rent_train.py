from real_estate.logging.logger import logging
from real_estate.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    logging.info("Builder floor rent training pipeline started")
    pipeline = TrainingPipeline()
    artifact = pipeline.run_bf_rent_training_pipeline()
    logging.info(f"Builder floor rent training pipeline finished: {artifact}")
    print(f"Model saved to: {artifact.model_file_path}")
    print(f"MAE: {artifact.mae:.0f} | MAPE: {artifact.mape*100:.1f}% | R2: {artifact.r2:.4f}")
