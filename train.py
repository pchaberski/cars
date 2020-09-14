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
from importlib import import_module


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

try:
    if CFG['loss_function'] == 'LabelSmoothingCrossEntropy':
        from models.label_smoothing_ce import LabelSmoothingCrossEntropy as _LOSS
    else:
        _LOSS = getattr(import_module('torch.nn'), CFG['loss_function']) 
    LOSS = _LOSS(**CFG['loss_params']) if CFG['loss_params'] is not None else _LOSS()
except:
    _LOSS = getattr(import_module('torch.nn'), 'CrossEntropyLoss')
    LOSS = _LOSS(**CFG['loss_params']) if CFG['loss_params'] is not None else _LOSS()
    LOGGER.warning(
        f"Invalid loss function: {CFG['loss_function']}\n"
        f'Running training with CrossEntropyLoss.'
    )
LOGGER.info(f'Setting loss function to {_LOSS.__name__}')

try:
    OPTIMIZER = getattr(import_module('torch.optim'), CFG['optimizer'])
    OPTIMIZER_PARAMS = CFG['optimizer_params']
except:
    OPTIMIZER = getattr(import_module('torch.optim'), 'SGD')
    OPTIMIZER_PARAMS = {'lr': 0.001}
    LOGGER.warning(
        f"Invalid optimizer: {CFG['optimizer']}\n"
        f'Running training with SGD with lr=0.001.'
    )
LOGGER.info(f'Setting optimizer to {OPTIMIZER.__name__}')

try:
    LR_SCHEDULER = getattr(import_module('torch.optim.lr_scheduler'), CFG['lr_scheduler'])
    LR_SCHEDULER_PARAMS = CFG['lr_scheduler_params']
except:
    LR_SCHEDULER = None
    LR_SCHEDULER_PARAMS = None
    LOGGER.warning(
        f"Invalid learning rate scheduler: {CFG['lr_scheduler']}\n"
        f'Running training without scheduler.'
    )


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
    if arch == 'ghost':
        base_model = arch_dict[arch](
            num_classes=196,
            img_channels=IMG_CHANNELS,
            dropout=CFG['dropout'],
            out_channels=CFG['out_channels']
        )
    else:
        base_model = arch_dict[arch](num_classes=196, img_channels=IMG_CHANNELS)

    # Init Stanford Cars Dataset Lightning Module
    data_module = StanfordCarsDataModule(
        data_path=DATA_PATH,
        batch_size=CFG['batch_size'],
        image_size=CFG['image_size'],
        convert_to_grayscale=CFG['convert_to_grayscale'],
        normalize=CFG['normalize'],
        normalization_params=CFG['normalization_params_grayscale'] if CFG['convert_to_grayscale'] else CFG['normalization_params_rgb'],
        augment_images=CFG['augment_images'],
        image_augmentations=CFG['image_augmentations'],
        augment_tensors=CFG['augment_tensors'],
        tensor_augmentations=CFG['tensor_augmentations']
    )

    # Init modeling Lightning Module
    model = NetModule(
        base_model,
        loss=LOSS,
        optimizer=OPTIMIZER,
        optimizer_params=OPTIMIZER_PARAMS,
        lr_scheduler=LR_SCHEDULER,
        lr_scheduler_params=LR_SCHEDULER_PARAMS
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
            'img_size': CFG['image_size'],
            'grayscale': CFG['convert_to_grayscale'],
            'normalize': CFG['normalize'],
            'norm_params_rgb': CFG['normalization_params_rgb'] if CFG['normalize'] and not CFG['convert_to_grayscale'] else None,
            'norm_params_gray': CFG['normalization_params_grayscale'] if CFG['normalize'] and CFG['convert_to_grayscale'] else None,
            'augment_images': CFG['augment_images'],
            'image_augmentations': CFG['image_augmentations'] if CFG['augment_images'] else None,
            'augment_tensors': CFG['augment_tensors'],
            'tensor_augmentations': CFG['tensor_augmentations'] if CFG['augment_tensors'] else None,
            'batch_size': CFG['batch_size'],
            'max_num_epochs': CFG['num_epochs'],
            'dropout': CFG['dropout'],
            'out_channels': CFG['out_channels'],
            'loss_function': _LOSS.__name__,
            'loss_params': CFG['loss_params'],
            'optimizer': OPTIMIZER.__name__,
            'learning_rate': OPTIMIZER_PARAMS['lr'],
            'weight_decay': OPTIMIZER_PARAMS['weight_decay'] if OPTIMIZER_PARAMS.get('weight_decay') is not None else None,
            'all_optimizer_params': OPTIMIZER_PARAMS,
            'lr_scheduler': LR_SCHEDULER.__name__ if LR_SCHEDULER is not None else None,
            'lr_scheduler_params': LR_SCHEDULER_PARAMS
        }

        neptune_logger = NeptuneLogger(
            api_key=CFG['neptune_api_token'],
            project_name='pchaberski/cars',
            experiment_name=CFG['architecture'] + '_' + RUN_TIMESTAMP,
            params=neptune_parameters
        )
    else:
        neptune_logger = None

    lr_monitor = pl.callbacks.LearningRateLogger()

    # Train
    trainer = pl.Trainer(
        max_epochs=CFG['num_epochs'],
        gpus=1,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        callbacks=[lr_monitor],
        logger=neptune_logger,
        num_sanity_val_steps=0
    )

    LOGGER.info(f'Running training...')
    trainer.fit(model, data_module)


    LOGGER.info('All done.')


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    run_training()
