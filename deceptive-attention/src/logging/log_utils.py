import datetime
import logging


def setup_logger(log_path, task, coeff, seed):
    """
    Sets up the logger to write onto the console, but also into a log file. File name contains date and time and main
    configurations for the respective run. The full model configuration is stored inside of the logfile.
    """

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                                      "%Y-%m-%d %H:%M:%S")
    root_logger = logging.getLogger()

    date_str = f"{datetime.datetime.now():%Y%m%d_%H%M}"
    file_name = f'{date_str}_task=%s_coeff=%s_seed=%s' % (task, coeff, seed)

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
