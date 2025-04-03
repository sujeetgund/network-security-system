import sys
from networksecurity.logging.logger import logger


def custom_error_message(error: Exception) -> str:
    _, _, exc_tb = sys.exc_info()
    if exc_tb is not None:
        filename = exc_tb.tb_frame.f_code.co_filename
        lineno = exc_tb.tb_lineno
        error_message = f"Error occurred in script: [{filename}] at line number: [{lineno}] error message: [{str(error)}]"
    else:
        error_message = f"Error occurred: [{str(error)}]"
    return error_message


class NetworkSecurityException(Exception):
    def __init__(self, error: Exception):
        self.error_message = custom_error_message(error)
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message
