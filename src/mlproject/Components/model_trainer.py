import os
import sys
import json
from dataclasses import dataclass

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import evaluate_models, save_object

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import dagshub
import mlflow
import mlflow.sklearn

dagshub.init(
    repo_owner='chaitanyarast23',
    repo_name='End-to-End-Scholarship_Eligibility_Prediction',
    mlflow=True
)


@dataclass
class ModelTrainerConfig:
    trained_model_config = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and test data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Decision Tree':       DecisionTreeClassifier(random_state=42),
                'Random Forest':       RandomForestClassifier(random_state=42),
                'XGBoost':             XGBClassifier(random_state=42, eval_metric='logloss'),
                'CatBoost':            CatBoostClassifier(random_state=42, verbose=0),
                'K-Neighbors':         KNeighborsClassifier()
            }

            params = {
                'Logistic Regression': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                },
                'Decision Tree': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Random Forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                'XGBoost': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'gamma': [0, 0.1, 0.3]
                },
                'CatBoost': {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5, 7]
                },
                'K-Neighbors': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            }

            # evaluate_models now returns F1 scores
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(model_report.values())
            best_model_name  = max(model_report, key=model_report.get)
            best_model       = models[best_model_name]

            logging.info(f"Best model: {best_model_name} | F1: {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No best model found — all models below F1 0.6", sys)

            
            predicted       = best_model.predict(X_test)
            predicted_proba = best_model.predict_proba(X_test)[:, 1]

            acc       = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted)
            recall    = recall_score(y_test, predicted)
            f1        = f1_score(y_test, predicted)
            roc_auc   = roc_auc_score(y_test, predicted_proba)

            
            with mlflow.start_run():

                # Params
                mlflow.log_param("model_name", best_model_name)
                mlflow.log_params(best_model.get_params())

                # Metrics
                mlflow.log_metric("accuracy",  acc)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall",    recall)
                mlflow.log_metric("f1_score",  f1)
                mlflow.log_metric("roc_auc",   roc_auc)

                # Confusion matrix as artifact
                cm = confusion_matrix(y_test, predicted).tolist()
                with open("confusion_matrix.json", "w") as f:
                    json.dump({"confusion_matrix": cm}, f, indent=2)
                mlflow.log_artifact("confusion_matrix.json")

                # Classification report as artifact
                report_str = classification_report(y_test, predicted)
                with open("classification_report.txt", "w") as f:
                    f.write(f"Model: {best_model_name}\n\n")
                    f.write(report_str)
                mlflow.log_artifact("classification_report.txt")

                # Model
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model"
                )

            logging.info(
                f"MLflow run logged | Acc: {acc:.4f} | P: {precision:.4f} | "
                f"R: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_config,
                obj=best_model
            )

            return acc

        except Exception as e:
            raise CustomException(e, sys)