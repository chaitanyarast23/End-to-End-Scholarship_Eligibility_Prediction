import sys
import numpy as np
import pandas as pd
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        self.model_path        = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Loading model and preprocessor")
            model        = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            logging.info("Transforming input features")
            transformed = preprocessor.transform(features)

            prediction = model.predict(transformed)
            proba      = model.predict_proba(transformed)[:, 1]

            return prediction, proba

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender           : str,
        nationality      : str,
        age              : int,
        english_grade    : float,
        math_grade       : float,
        sciences_grade   : float,
        language_grade   : float,
        portfolio_rating : float,
        coverletter_rating: float,
        refletter_rating : float,
    ):
        self.gender            = gender
        self.nationality       = nationality
        self.age               = age
        self.english_grade     = english_grade
        self.math_grade        = math_grade
        self.sciences_grade    = sciences_grade
        self.language_grade    = language_grade
        self.portfolio_rating  = portfolio_rating
        self.coverletter_rating = coverletter_rating
        self.refletter_rating  = refletter_rating

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                "gender"            : [self.gender],
                "nationality"       : [self.nationality],
                "age"               : [self.age],
                "english.grade"     : [self.english_grade],
                "math.grade"        : [self.math_grade],
                "sciences.grade"    : [self.sciences_grade],
                "language.grade"    : [self.language_grade],
                "portfolio.rating"  : [self.portfolio_rating],
                "coverletter.rating": [self.coverletter_rating],
                "refletter.rating"  : [self.refletter_rating],
            }
            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)