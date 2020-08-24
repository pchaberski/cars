"""Run training procedure."""


from utils.logger import configure_logger
from utils.configuration import load_config
from utils.timer import timer
from models.squeezenet import SqueezeNet_10, SqueezeNet_11
from models.squeezenext import (
    SqueezeNext23_10, SqueezeNext23_10_v5,
    SqueezeNext23_20, SqueezeNext23_20_v5)
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
    if CFG['architecture'] == 'sq_10':
        LOGGER.info('Setting architecture to SqueezeNet_10')
        base_model = SqueezeNet_10(num_classes=196)
    elif CFG['architecture'] == 'sq_11':
        LOGGER.info('Setting architecture to SqueezeNet_11')
        base_model = SqueezeNet_11(num_classes=196)
    elif CFG['architecture'] == 'sqnxt_1':
        LOGGER.info('Setting architecture to SqueezeNext23_10')
        base_model = SqueezeNext23_10(num_classes=196)
    elif CFG['architecture'] == 'sqnxt_1v5':
        LOGGER.info('Setting architecture to SqueezeNext23_10_v5')
        base_model = SqueezeNext23_10_v5(num_classes=196)
    elif CFG['architecture'] == 'sqnxt_2':
        LOGGER.info('Setting architecture to SqueezeNext23_20')
        base_model = SqueezeNext23_20(num_classes=196)
    elif CFG['architecture'] == 'sqnxt_2v5':
        LOGGER.info('Setting architecture to SqueezeNext23_20_v5')
        base_model = SqueezeNext23_20_v5(num_classes=196)
    else:
        LOGGER.warn('Invalid architecture; setting to default.')
        LOGGER.info('Setting architecture to SqueezeNet_10')
        base_model = SqueezeNet_10(num_classes=196)

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

    LOGGER.info(f"Running training with: {CFG['architecture']}")
    trainer.fit(model)

    # Test
    trainer.test(model)

    LOGGER.info('All done.')


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    run_training()
