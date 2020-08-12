from utils.logger import configure_logger
from utils.configuration import load_config


def main():
    cfg = load_config('config.yml')
    logger = configure_logger(__name__, cfg['logging_dir'], cfg['loglevel'])

    logger.info('Done')


if __name__ == '__main__':
    main()
