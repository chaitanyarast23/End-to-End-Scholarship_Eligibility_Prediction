import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import read_sql_query
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class Data_ingestion_config:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

class Data_Ingestion:
    def __init__(self):
        self.data_ingestion_config=Data_ingestion_config()

    def initate_Data_ingestion(self):
        try:
            df=read_sql_query()
            logging.info("Reading from MySQL")
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)

            train_df,test_df=train_test_split(df, test_size=0.2,random_state=42)

            train_df.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_df.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion is Completed")

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)