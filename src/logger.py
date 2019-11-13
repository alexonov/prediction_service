import logging
import logging.handlers
import os
from datetime import datetime
import sys

# Logging Levels
# CRITICAL  50
# ERROR 40
# WARNING   30
# INFO  20
# DEBUG 10
# NOTSET    0


def set_up_logging(use_file=False):

    logger = logging.getLogger(__name__)
    format = '[%(asctime)s] [%(levelname)s] [%(message)s] [--> %(pathname)s [%(process)d]:]'

    if use_file:
        file_path = sys.modules[__name__].__file__
        project_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
        log_location = project_path + '/logs/'
        if not os.path.exists(log_location):
            os.makedirs(log_location)

        current_time = datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        file_name = current_date + '.log'
        file_location = log_location + file_name
        with open(file_location, 'a+'):
            pass

        # To store in file
        logging.basicConfig(format=format, filemode='a+', filename=file_location, level=logging.DEBUG)
    else:
        # To print only
        logging.basicConfig(format=format, level=logging.DEBUG)

    return logger


logger = set_up_logging()