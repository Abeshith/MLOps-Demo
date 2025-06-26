import os
import sys


def error_message_detail(error, error_details:sys):
    _, _, exc_tb = error_details.exc_info()
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    error_message = f"Error occurred in script: [{file_name}] at line number: [{exc_tb.tb_lineno}] with error message: [{str(error)}]"
    return error_message


class CustomException(Exception):
    def __init__(self, error, error_details:sys):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_details)

    def __str__(self):
        return self.error_message
    

if __name__ == "__main__":
    try:
        1 / 0
    except Exception as e:
        raise CustomException(e, sys) from e
    finally:
        print("This will always execute, even if an exception occurs.")