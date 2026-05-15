from src.mlproject.logger import logging 
from src.mlproject.exception import CustomException
import sys


if __name__=="__main__":
    logging.info("The exicution is Started")

    try:
        a=10/0

    except Exception as e :
        logging.info("CustomException Occur")
        raise CustomException(e,sys)

