"""Pytorch Lightning module for Stanford Cars Dataset."""


from utils.configuration import load_config
import pytorch_lightning as pl
from datasets.stanford_data import StanfordCarsDataset
import numpy as np
from torch.utils.data import DataLoader, random_split
import os
import torch
import numpy as np


class StanfordCarsDataModule(pl.LightningDataModule):

    def __init__(
        self, data_path, batch_size,
        image_size=[227, 227],
        convert_to_grayscale=False, normalize=False,
        normalization_params={'mean': None, 'std': None},
        augment_images=False,
        image_augmentations=None,
        augment_tensors=False,
        tensor_augmentations=None
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.convert_to_grayscale = convert_to_grayscale
        self.normalize = normalize
        self.normalization_params = normalization_params
        self.augment_images = augment_images
        self.image_augmentations = image_augmentations
        self.augment_tensors = augment_tensors
        self.tensor_augmentations = tensor_augmentations

    def setup(self, stage='fit'):

        self.data_train = StanfordCarsDataset(
            data_path=os.path.join(self.data_path, 'train'),
            labels_fpath=os.path.join(self.data_path, 'train_labels.csv'),
            image_size=self.image_size,
            convert_to_grayscale=self.convert_to_grayscale,
            normalize=self.normalize,
            normalization_params=self.normalization_params,
            augment_images=self.augment_images,
            image_augmentations=self.image_augmentations,
            augment_tensors=self.augment_tensors,
            tensor_augmentations=self.tensor_augmentations
        )

        self.data_valid = StanfordCarsDataset(
            data_path=os.path.join(self.data_path, 'test'),
            labels_fpath=os.path.join(self.data_path, 'test_labels.csv'),
            image_size=self.image_size,
            convert_to_grayscale=self.convert_to_grayscale,
            normalize=self.normalize,
            normalization_params=self.normalization_params,
            augment_images=False,
            augment_tensors=False
        )

    def train_dataloader(self):
        loader = DataLoader(self.data_train, batch_size=self.batch_size, num_workers=4, pin_memory=True, shuffle=True)

        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data_valid, num_workers=4, pin_memory=True, batch_size=self.batch_size)

        return loader
