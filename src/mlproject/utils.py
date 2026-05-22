import os
import sys
import pandas as pd
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pymysql
from dotenv import load_dotenv

load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")

def read_sql_query():
    try:
        logging.info("Start Reading SQL Database")
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Establised",mydb)
        df=pd.read_sql_query('Select * from student',mydb)
        print(df.head())

        return df


    except Exception as e:
        raise CustomException(e, sys)
    
    
