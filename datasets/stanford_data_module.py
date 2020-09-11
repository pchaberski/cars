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
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.convert_to_grayscale = convert_to_grayscale
        self.normalize = normalize
        self.normalization_params = normalization_params

    def setup(self, stage):

        self.data_train = StanfordCarsDataset(
            data_path=os.path.join(self.data_path, 'train'),
            labels_fpath=os.path.join(self.data_path, 'train_labels.csv'),
            convert_to_grayscale=self.convert_to_grayscale,
            normalize=self.normalize,
            normalization_params=self.normalization_params,
            image_size=self.image_size)

        self.data_valid = StanfordCarsDataset(
            data_path=os.path.join(self.data_path, 'test'),
            labels_fpath=os.path.join(self.data_path, 'test_labels.csv'),
            convert_to_grayscale=self.convert_to_grayscale,
            normalize=self.normalize,
            normalization_params=self.normalization_params,
            image_size=self.image_size)

    def train_dataloader(self):
        loader = DataLoader(self.data_train, batch_size=self.batch_size, num_workers=4, pin_memory=True, shuffle=True)

        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data_valid, num_workers=4, pin_memory=True, batch_size=self.batch_size)

        return loader
