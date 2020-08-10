"""Configuration utilities"""


import yaml
import os


def load_config(fpath):
    """
    Load configuration file as dict.

    :param fpath: file path to config.yml
    :type fpath: string
    :return: dictionary with config settings
    :rtype: dict
    """
    assert os.path.isfile(fpath), 'File does not exist'

    with open(fpath, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    return cfg
