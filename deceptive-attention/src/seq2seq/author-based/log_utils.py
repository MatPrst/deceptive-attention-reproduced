import datetime
import logging


def setup_logger(log_path, file_name_suffix, detailed=False):
    """
    Sets up the logger to write onto the console, but also into a log file. File name contains date and time and main
    configurations for the respective run. The full model configuration is stored inside of the logfile.
    """

    format = "%(asctime)s  [%(levelname)-5.5s]  %(message)s" if detailed else "%(message)s"
    log_formatter = logging.Formatter(format, "%Y-%m-%d %H:%M:%S")
    root_logger = logging.getLogger()
    root_logger.handlers[:] = []

    file_name = f'{datetime.datetime.now():%Y%m%d_%H%M}_{file_name_suffix}'
    file_handler = logging.FileHandler("{0}/{1}.log".format(log_path, file_name))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

    return root_logger
