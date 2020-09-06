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
import neptune
from pytorch_lightning.loggers import NeptuneLogger


CFG = load_config('config.yml')
LOGGER = configure_logger(__name__, CFG['logging_dir'], CFG['loglevel'])
OUTPUT_PATH = os.path.join(os.getcwd(), CFG['output_path'])
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d%H%M%S')


try:
    import google.colab
    RUNTIME = 'colab'
except:
    RUNTIME = 'local'


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

IMG_CHANNELS = 1 if CFG['convert_to_grayscale'] else 3


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
    base_model = arch_dict[arch](num_classes=196, img_channels=IMG_CHANNELS)

    optim = CFG['optimizer']
    assert optim in ['adam', 'sgd'], (
        f'Optimizer has to be one of: `adam`, `sgd`'
        f'Provided optimizer: {optim}'
    )

    LOGGER.info(f'Setting optimizer to: {optim}')

    # Init Stanford Cars Dataset Lightning Module
    data_module = StanfordCarsDataModule(
        data_path=DATA_PATH,
        batch_size=CFG['batch_size'],
        validate_on_test=CFG['validate_on_test'], split_ratios=CFG['split_ratios'],
        convert_to_grayscale=CFG['convert_to_grayscale'],
        normalize=CFG['normalize'],
        normalization_params=CFG['normalization_params_grayscale'] if CFG['convert_to_grayscale'] else CFG['normalization_params_rgb'],
        image_size=CFG['image_size'], seed=CFG['seed']
    )

    # Init modeling Lightning Module
    model = NetModule(
        base_model, optimizer=optim, learning_rate=CFG['learning_rate'],
        validate_on_test=CFG['validate_on_test']
    )

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

    # Neptune logger
    if CFG['log_to_neptune']:
        neptune_parameters = {
            'runtime': RUNTIME,
            'architecture': arch_dict[arch].__name__,
            'num_params': sum(p.numel() for p in base_model.parameters() if p.requires_grad),
            'grayscale': CFG['convert_to_grayscale'],
            'normalize': CFG['normalize'],
            'norm_params_rgb': CFG['normalization_params_rgb'] if CFG['normalize'] and not CFG['convert_to_grayscale'] else None,
            'norm_params_gray': CFG['normalization_params_grayscale'] if CFG['normalize'] and CFG['convert_to_grayscale'] else None,
            'batch_size': CFG['batch_size'],
            'validate_on_test': CFG['validate_on_test'],
            'train_valid_split': CFG['split_ratios'] if not CFG['validate_on_test'] else None,
            'max_num_epochs': CFG['num_epochs'],
            'optimizer': CFG['optimizer'],
            'learning_rate': CFG['learning_rate'],
            'random_seed': CFG['seed']
        }

        neptune_logger = NeptuneLogger(
            api_key=CFG['neptune_api_token'],
            project_name='pchaberski/cars',
            experiment_name=CFG['architecture'] + '_' + RUN_TIMESTAMP,
            params=neptune_parameters
        )
    else:
        neptune_logger = None

    # Train
    trainer = pl.Trainer(
        max_epochs=CFG['num_epochs'],
        gpus=1,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        logger=neptune_logger
    )

    LOGGER.info(f'Running training with: {arch}')
    trainer.fit(model, data_module)

    # Test
    if not CFG['validate_on_test']:
        trainer.test(model, datamodule=data_module)

    LOGGER.info('All done.')


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    run_training()
