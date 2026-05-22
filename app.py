from src.mlproject.logger import logging 
from src.mlproject.exception import CustomException
from src.mlproject.Components.data_ingestion import Data_Ingestion
import sys


if __name__=="__main__":
    logging.info("The exicution is Started")

    try:
        data_ingestion=Data_Ingestion()
        data_ingestion.initate_Data_ingestion()

    except Exception as e :
        logging.info("CustomException Occur")
        raise CustomException(e,sys)

