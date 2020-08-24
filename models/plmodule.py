"""Pytorch Lightning Module for Stanford Cars Dataset."""


from utils.configuration import load_config
import pytorch_lightning as pl
from datasets.stanford import StanfordCarsDataset
import numpy as np
from torch.utils.data import DataLoader, random_split
import os
import torch
from torch import nn
from pytorch_lightning.metrics.functional import accuracy


CFG = load_config('config.yml')


class StanfordLightningModule(pl.LightningModule):

    def __init__(self, base_model, data_path, batch_size, image_size, split_ratios):
        super().__init__()
        self.base_model = base_model
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.split_ratios = split_ratios
        self.loss = nn.CrossEntropyLoss()

    @pl.core.decorators.auto_move_data
    def forward(self, input):
        return self.base_model.forward(input)

    def prepare_data(self):
        assert round(sum(self.split_ratios), 5) == 1., \
            'Split ratios has to sum up to 1.'

        data = StanfordCarsDataset(
            data_path=os.path.join(CFG['stanford_data_path'], 'train'),
            labels_fpath=os.path.join(CFG['stanford_data_path'], 'train_labels.csv'),
            image_size=self.image_size)

        split_sizes = (len(data) * np.array(self.split_ratios)).astype(np.int)
        split_sizes[-1] = split_sizes[-1] + (len(data) - sum(split_sizes))

        self.data_train, self.data_valid, self.data_test = \
            random_split(
                data, split_sizes.tolist(),
                generator=torch.Generator().manual_seed(CFG['seed']))

    def step(self, batch, batch_idx, loss_type):
        input, labels = batch

        preds = self.forward(input)
        pred_classes = torch.argmax(preds, dim=1)

        loss = self.loss(preds, labels)
        logs = {loss_type: loss}

        # Metrics
        logs['accuracy'] = accuracy(pred_classes, labels)

        return {loss_type: loss, 'log': logs}

    def training_step(self, batch_train, batch_idx):
        return self.step(batch_train, batch_idx, 'loss')

    def validation_step(self, batch_valid, batch_idx):
        return self.step(batch_valid, batch_idx, 'val_loss')

    def test_step(self, batch_test, batch_idx):
        return self.step(batch_test, batch_idx, 'test_loss')

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([output['test_loss'] for output in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2)
