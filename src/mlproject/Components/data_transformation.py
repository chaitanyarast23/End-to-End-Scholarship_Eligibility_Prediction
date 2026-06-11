import os
import sys

from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        """
        This function is responsible for
        data transformation
        """

        try:

            numerical_columns = [
                'age',
                'english.grade',
                'math.grade',
                'sciences.grade',
                'language.grade',
                'portfolio.rating',
                'coverletter.rating',
                'refletter.rating'
            ]

            categorical_columns = [
                'gender',
                'nationality'
            ]

            logging.info(
                f"Numerical Columns: {numerical_columns}"
            )

            logging.info(
                f"Categorical Columns: {categorical_columns}"
            )

            # Numerical Pipeline

            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy='median')
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # Categorical Pipeline

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),

                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown='ignore')
                    ),

                    (
                        "scaler",
                        StandardScaler(with_mean=False)
                    )
                ]
            )

            logging.info(
                "Numerical and Categorical Pipelines Created"
            )

            preprocessor = ColumnTransformer(
                [
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),

                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(
                "Train and Test Data Read Successfully"
            )

            # Drop unnecessary columns

            train_df = train_df.drop(columns=['name'])
            test_df = test_df.drop(columns=['name'])

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "scholarship_eligible"

            # Training Data

            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )

            target_feature_train_df = train_df[
                target_column_name
            ]

            # Testing Data

            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )

            target_feature_test_df = test_df[
                target_column_name
            ]

            logging.info(
                "Applying preprocessing object on training and testing dataframes"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            input_feature_train_arr = input_feature_train_arr.toarray()
            input_feature_test_arr = input_feature_test_arr.toarray()

            

            train_arr = np.hstack((
                input_feature_train_arr,
                np.array(target_feature_train_df).reshape(-1, 1)
            ))

            test_arr = np.hstack((
                input_feature_test_arr,
                np.array(target_feature_test_df).reshape(-1, 1)
            ))

            

            logging.info(
                "Preprocessing Completed"
            )

            save_object(

                file_path=self.data_transformation_config.preprocess_obj_file_path,

                obj=preprocessing_obj

            )

            logging.info(
                "Preprocessing Object Saved Successfully"
            )

            return (

                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_file_path

            )

        except Exception as e:
            raise CustomException(e, sys)