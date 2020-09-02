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

    def __init__(self, data_path, batch_size, image_size, split_ratios, seed):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.split_ratios = split_ratios
        self.seed = seed

    def setup(self, stage):
        assert round(sum(self.split_ratios), 5) == 1., \
            'Split ratios has to sum up to 1.'

        if stage == 'fit':
            data_trainvalid = StanfordCarsDataset(
                data_path=os.path.join(self.data_path, 'train'),
                labels_fpath=os.path.join(self.data_path, 'train_labels.csv'),
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
                image_size=self.image_size)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)
