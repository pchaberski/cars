"""Run training procedure."""


from utils.logger import configure_logger
from utils.configuration import load_config
from utils.timer import timer
from models.arch_dict import get_architectures_dictionary
from datasets.stanford_data_module import StanfordCarsDataModule
from models.net_module import NetModule
from datetime import datetime
import torch
import pytorch_lightning as pl
import os
import sys
import getopt


CFG = load_config('config.yml')
LOGGER = configure_logger(__name__, CFG['logging_dir'], CFG['loglevel'])
OUTPUT_PATH = os.path.join(os.getcwd(), CFG['output_path'])
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')


try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:')
    if len(opts) == 0 or len(opts) > 1:
        DATA_PATH = CFG['stanford_data_path']
    else:
        LOGGER.warning('Setting alternative data path')
        DATA_PATH = opts[0][1]
except getopt.GetoptError:
    LOGGER.warning('Argument parsing error! [usage: train.py -d <alternative_data_path>]')
    LOGGER.warning('Setting default data path')
    DATA_PATH = CFG['stanford_data_path']

LOGGER.info(f'Data path: {DATA_PATH}')


def run_training():
    """Perform training."""
    arch = CFG['architecture']
    arch_dict = get_architectures_dictionary()

    assert arch in arch_dict.keys(), (
        f'Architecture name has to be one of:\n'
        f'{list(arch_dict.keys())}\n'
        f'Provided architecture: {arch}'
    )

    LOGGER.info(f'Setting architecture to {arch_dict[arch].__name__}')
    base_model = arch_dict[arch](num_classes=196)

    optim = CFG['optimizer']
    assert optim in ['adam', 'sgd'], (
        f'Optimizer has to be one of: `adam`, `sgd`'
        f'Provided architecture: {optim}'
    )

    LOGGER.info(f'Set optimizer to: {optim}')

    # Init Stanford Cars Dataset Lightning Module
    data_module = StanfordCarsDataModule(
        data_path=DATA_PATH,
        batch_size=CFG['batch_size'], image_size=(227, 227),
        split_ratios=CFG['split_ratios'], seed=CFG['seed']
    )

    # Init modeling Lightning Module
    model = NetModule(
        base_model, optimizer=optim, learning_rate=CFG['learning_rate'])

    # Callbacks
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=10,
        verbose=False,
        mode='min'
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=OUTPUT_PATH,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=CFG['architecture'] + '_' + RUN_TIMESTAMP + '_'
    )

    # Train
    trainer = pl.Trainer(
        max_epochs=CFG['num_epochs'],
        gpus=1,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        fast_dev_run=False
    )

    LOGGER.info(f'Running training with: {arch}')
    trainer.fit(model, data_module)

    # Test
    trainer.test(model, datamodule=data_module)

    LOGGER.info('All done.')


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    run_training()
