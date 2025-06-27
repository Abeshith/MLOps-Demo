import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from source.logger import logging
from source.exception import CustomException

from sklearn.metrics import  r2_score


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
    


def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)

            report[list(models.keys())[i]] = r2


        return report
    except Exception as e:
        logging.error("An error occurred while evaluating the model.")
        raise CustomException(e, sys) from e
    