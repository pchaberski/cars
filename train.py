"""Run training procedure."""


from utils.logger import configure_logger
from utils.configuration import load_config
from utils.timer import timer
from models.arch_dict import get_architectures_dictionary
from datasets.stanford import StanfordCarsDataset
from models.plmodule import StanfordLightningModule
import os
import torch
import pytorch_lightning as pl


CFG = load_config('config.yml')
LOGGER = configure_logger(__name__, CFG['logging_dir'], CFG['loglevel'])
OUTPUT_PATH = os.path.join(os.getcwd(), CFG['output_path'])


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

    # Init Lightning Module
    model = StanfordLightningModule(
        base_model, data_path=CFG['stanford_data_path'],
        batch_size=CFG['batch_size'], image_size=(227, 227),
        split_ratios=CFG['split_ratios'])

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
        prefix=CFG['architecture'] + '_'
    )

    # Train
    trainer = pl.Trainer(
        max_epochs=CFG['num_epochs'], gpus=1,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback
        )

    LOGGER.info(f'Running training with: {arch})
    trainer.fit(model)

    # Test
    trainer.test(model)

    LOGGER.info('All done.')


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    run_training()
