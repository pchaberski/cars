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
        validate_on_test=False, split_ratios=[0.8, 0.2],
        convert_to_grayscale=False, normalize=False,
        normalization_params={'mean': None, 'std': None},
        image_size=[227, 227],
        seed=42
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.validate_on_test = validate_on_test
        self.split_ratios = split_ratios
        self.convert_to_grayscale = convert_to_grayscale
        self.normalize = normalize
        self.normalization_params = normalization_params
        self.image_size = image_size
        self.seed = seed

    def setup(self, stage):
        assert round(sum(self.split_ratios), 5) == 1., \
            'Split ratios has to sum up to 1.'

        if not self.validate_on_test:
            if stage == 'fit':
                data_trainvalid = StanfordCarsDataset(
                    data_path=os.path.join(self.data_path, 'train'),
                    labels_fpath=os.path.join(self.data_path, 'train_labels.csv'),
                    convert_to_grayscale=self.convert_to_grayscale,
                    normalize=self.normalize,
                    normalization_params=self.normalization_params,
                    image_size=self.image_size)

                split_sizes = (len(data_trainvalid) * np.array(self.split_ratios)).astype(np.int).tolist()
                split_sizes[-1] = split_sizes[-1] + (len(data_trainvalid) - sum(split_sizes))

                self.data_train, self.data_valid = random_split(
                    data_trainvalid, split_sizes,
                    generator=torch.Generator().manual_seed(self.seed)
                )

            if stage == 'test':
                self.data_test = StanfordCarsDataset(
                    data_path=os.path.join(self.data_path, 'test'),
                    labels_fpath=os.path.join(self.data_path, 'test_labels.csv'),
                    convert_to_grayscale=self.convert_to_grayscale,
                    normalize=self.normalize,
                    normalization_params=self.normalization_params,
                    image_size=self.image_size)
        else:
            if stage == 'fit':
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

    def test_dataloader(self):
        loader = DataLoader(self.data_test, num_workers=4, pin_memory=True, batch_size=self.batch_size) \
            if not self.validate_on_test else None

        return loader
