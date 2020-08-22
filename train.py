"""Run training procedure."""


from utils.logger import configure_logger
from utils.configuration import load_config
from utils.timer import timer
from datasets.stanford import StanfordCarsDataset
from models.squeezenet import SqueezeNet_10, SqueezeNet_11
from models.squeezenext import (
    SqueezeNext23_10, SqueezeNext23_10_v5,
    SqueezeNext23_20, SqueezeNext23_20_v5)
from models.plmodule import StanfordLightningModule

import os
import torch


CFG = load_config('config.yml')
LOGGER = configure_logger(__name__, CFG['logging_dir'], CFG['loglevel'])


def run_training():
    """Perform training."""

    data = StanfordCarsDataset(
        data_path=os.path.join(CFG['stanford_data_path'], 'train'),
        labels_fpath=os.path.join(CFG['stanford_data_path'], 'train_labels.csv'),
        image_size=(227, 227))

    # DEBUG
    # Sample input
    sample_input = data[0][0]
    sample_batch = sample_input.unsqueeze(0)
    print(f'Input shape: {sample_batch.shape}')
    # /DEBUG

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

    model = StanfordLightningModule(base_model)

    # DEBUG
    # Use model on sample batch
    with torch.no_grad():
        output = model.forward(sample_batch)
    print(f'Prediction: {torch.argmax(torch.nn.functional.softmax(output[0], dim=0))}')
    # /DEBUG

    LOGGER.info('All done.')


if __name__ == '__main__':
    run_training()
