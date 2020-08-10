"""Logging utilities"""


import logging
import os
from datetime import datetime


def configure_logger(name, logging_dir, loglevel='INFO'):
    """

    :param name: scope name
    :type name: string
    :param logging_dir: directory where log files are stored
    :type logging_dir: string
    :param loglevel: logging level
    :type: string
    :return: logger
    :rtype: logging.Logger
    """
    assert loglevel in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    today = datetime.today().strftime('%Y%m%d')
    log_fname = f'{today}.log'
    pid = os.getpid()

    file_handler = logging.FileHandler(os.path.join(logging_dir, log_fname))
    file_handler.setLevel(loglevel)
    file_formatter = logging.Formatter(f'%(asctime)s: %(name)s: %(levelname)s: PID[{pid}]: %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(loglevel)
    console_formatter = logging.Formatter(f'%(asctime)s: %(name)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    assert isinstance(logger, logging.Logger), 'Failed to create logger'

    return logger