from source.exception import CustomException
from source.logger import logging
import sys

logging.info("This is an info message from the data transformation")
try:
    a = 1 / 0
except Exception as e:
    logging.error("An error occurred in the main block.")
    raise CustomException(e, sys) from e