from src.mlproject.pipelines.training_pipeline import TrainingPipeline
from src.mlproject.logger import logging
import sys

if __name__ == "__main__":
    logging.info("Manual training started")
    pipeline = TrainingPipeline()
    f1 = pipeline.run()
    print(f"\n✅ Training complete — Best F1 Score: {f1:.4f}")