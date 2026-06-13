import os
import sys
import pandas as pd
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pymysql
from dotenv import load_dotenv
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def read_sql_query():
    try:
        logging.info("Start Reading SQL Database")
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info(f"Connection Established: {mydb}")
        df = pd.read_sql_query('SELECT * FROM student_data', mydb)
        print(df.head())
        return df

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}  # stores f1 score per model
        train_report = {}  # stores train f1 to detect overfitting

        for model_name, model in models.items():
            para = param[model_name]

            # GridSearchCV with F1 scoring (better than accuracy for imbalanced data)
            gs = GridSearchCV(
                model, para,
                cv=3,
                scoring='f1',       
                n_jobs=-1,          
                verbose=0
            )
            gs.fit(X_train, y_train)

            # Refit with best params
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_f1 = f1_score(y_train, y_train_pred)
            test_f1  = f1_score(y_test, y_test_pred)

            report[model_name] = test_f1
            train_report[model_name] = train_f1

            # ✅ Overfitting warning
            gap = train_f1 - test_f1
            if gap > 0.1:
                logging.warning(
                    f"[{model_name}] Possible overfitting — "
                    f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}, Gap: {gap:.4f}"
                )
            else:
                logging.info(
                    f"[{model_name}] Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f}"
                )

        return report

    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)