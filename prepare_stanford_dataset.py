"""Prepare Stanford dataset."""


from datasets.stanford_utils import prepare_stanford_dataset
from utils.logger import configure_logger
from utils.configuration import load_config


CFG = load_config('config.yml')
LOGGER = configure_logger(__name__, CFG['logging_dir'], CFG['loglevel'])


if __name__ == '__main__':
    LOGGER.info('Started data preparation...')
    prepare_stanford_dataset(
        CFG['stanford_raw_data_path'],
        CFG['stanford_data_path'],
        get_time=True, logger=LOGGER
    )
    LOGGER.info('All done.')
