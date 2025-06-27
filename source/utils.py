import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from source.logger import logging
from source.exception import CustomException

from sklearn.metrics import  r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    
    Args:
        file_path (str): The path where the object will be saved.
        obj (object): The object to be saved.
    """
    try:
        import pickle
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

    except Exception as e:
        logging.error("Error occurred while saving the object.")
        raise CustomException(e, sys) from e
    


def save_model_report(file_path, model_report):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'w') as report_file:
            for model_name, score in model_report.items():
                report_file.write(f"{model_name}: {score}\n")
        
        logging.info(f"Model report saved successfully to {file_path}")
                
    except Exception as e:
        logging.error("Error occurred while saving the model report.")
        raise CustomException(e, sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        logging.info("Starting model evaluation and hyperparameter tuning...")

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            model_params = params[model_name]

            logging.info(f"Training model: {model_name}")
            logging.info(f"Hyperparameter search space: {model_params}")

            if model_params:  # Only perform GridSearch if parameters are provided
                cv = GridSearchCV(
                    estimator=model,
                    param_grid=model_params,
                    cv=3,
                    n_jobs=-1,
                    verbose=2
                )
                cv.fit(X_train, y_train)
                
                logging.info(f"Best parameters for {model_name}: {cv.best_params_}")
                logging.info(f"Best cross-validation score for {model_name}: {cv.best_score_:.4f}")
                
                model.set_params(**cv.best_params_)
            else:
                logging.info(f"No hyperparameters to tune for {model_name}")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)

            logging.info(f"Final R2 score for {model_name}: {r2:.4f}")
            logging.info("-" * 50)

            report[model_name] = r2

        logging.info("Model evaluation completed successfully!")
        return report
    except Exception as e:
        logging.error("An error occurred while evaluating the model.")
        raise CustomException(e, sys) from e
