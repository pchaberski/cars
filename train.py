from utils.logger import configure_logger
from utils.configuration import load_config


if __name__=='__main__':
    cfg = load_config('config.yml')
    logger = configure_logger(__name__, cfg['logging_dir'], cfg['loglevel'])

    logger.info('Done')

