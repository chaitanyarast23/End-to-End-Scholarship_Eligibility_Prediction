from src.mlproject.logger import logging 
from src.mlproject.exception import CustomException
from src.mlproject.Components.data_ingestion import Data_Ingestion
from src.mlproject.Components.data_transformation import DataTransformation
from src.mlproject.Components.model_trainer import ModelTrainer
import sys


if __name__=="__main__":
    logging.info("The exicution is Started")

    try:
        data_ingestion=Data_Ingestion()
        train_path,test_path=data_ingestion.initate_Data_ingestion()

        data_transformation=DataTransformation()
        train_arr, test_arr,preprocess_path=data_transformation.initiate_data_transformation(train_path,test_path)

        model_trainer=ModelTrainer()
        x=model_trainer.initiate_model_trainer(train_arr,test_arr)
        print(x)

    except Exception as e :
        logging.info("CustomException Occur")
        raise CustomException(e,sys)

