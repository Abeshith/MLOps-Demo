import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from source.logger import logging
from source.exception import CustomException

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