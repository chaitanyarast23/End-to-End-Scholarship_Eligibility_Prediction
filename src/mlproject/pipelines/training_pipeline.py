from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.Components.data_ingestion import Data_Ingestion
from src.mlproject.Components.data_transformation import DataTransformation
from src.mlproject.Components.model_trainer import ModelTrainer
import sys


class TrainingPipeline:
    def run(self):
        try:
            logging.info("===== Training Pipeline Started =====")

            # Step 1: Data Ingestion
            logging.info("Step 1: Data Ingestion")
            data_ingestion = Data_Ingestion()
            train_path, test_path = data_ingestion.initate_Data_ingestion()

            # Step 2: Data Transformation
            logging.info("Step 2: Data Transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocess_path = data_transformation.initiate_data_transformation(
                train_path, test_path
            )

            # Step 3: Model Training
            logging.info("Step 3: Model Training")
            model_trainer = ModelTrainer()
            f1_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"===== Training Pipeline Completed | Best F1: {f1_score:.4f} =====")
            return f1_score

        except Exception as e:
            logging.error("Training Pipeline Failed")
            raise CustomException(e, sys)