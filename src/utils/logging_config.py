import os
import logging
import datetime
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get('ROOT_PATH')

def get_logger(logger_id, current_date=datetime.datetime.now().strftime('%Y%m%d')):

    logging.getLogger(logger_id).handlers.clear()

    # Log file name
    log_file = os.path.join(ROOT_PATH, f'log/{current_date}_log.txt')
    err_file = os.path.join(ROOT_PATH, f'log/{current_date}_error.txt')

    # Create a logger instance 
    logger = logging.getLogger(logger_id)
    logger.setLevel(logging.INFO)

    # Create file handler that logs INFO and higher level messages
    info_handler = logging.FileHandler(log_file)
    info_handler.setLevel(logging.INFO)

    # Create file handler that logs ERROR and higher level messages
    error_handler = logging.FileHandler(err_file)
    error_handler.setLevel(logging.ERROR)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] : %(message)s", "%Y-%m-%d %H:%M:%S %p")
    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    # Add the handlers to logger
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    return logger