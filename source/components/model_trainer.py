import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from source.logger import logging
from source.exception import CustomException
from source.utils import save_object, evaluate_models
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.linear_model import (
    LinearRegression, 
    Ridge,
    Lasso
)
from sklearn.metrics import ( 
    r2_score, 
    mean_absolute_error, 
    mean_squared_error
)

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    model_report_file_path: str = os.path.join('artifacts', 'model_report.txt')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Entering the model trainer component")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "SVR": SVR(),
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso()
            }

            model_report : dict = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models
            )

            # Filter only R2 scores to find the best model
            r2_scores = {key: value for key, value in model_report.items() if key.endswith('_r2_score')}
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy.", sys)
            
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model training completed and model saved successfully.")

            with open(self.model_trainer_config.model_report_file_path, 'w') as report_file:
                for model_name, score in model_report.items():
                    report_file.write(f"{model_name}: {score}\n")
                    
            logging.info("Model report saved successfully.")

            predictions = best_model.predict(X_test)

            r2 = r2_score(y_test, predictions)

            logging.info(f"Model evaluation metrics - R2: {r2}")

            return r2
            


        except Exception as e:
            logging.error("An error occurred in the model trainer component.")
            raise CustomException(e, sys) from e