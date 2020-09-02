"""Pytorch Lightning module for modeling."""


from utils.configuration import load_config
import pytorch_lightning as pl
import torch
from torch import nn
# from pytorch_lightning.metrics.functional import accuracy
from torch.optim import SGD, Adam


class NetModule(pl.LightningModule):

    def __init__(self, base_model, optimizer, learning_rate):
        super().__init__()
        self.base_model = base_model
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = float(learning_rate)
        self.optimizer = Adam if optimizer == 'adam' else SGD

    def forward(self, input):
        return self.base_model(input)

    def step(self, batch, batch_idx, loss_type):
        input, labels = batch

        preds = self(input)
        pred_classes = torch.argmax(preds, dim=1)

        loss = self.loss(preds, labels)
        logs = {loss_type: loss}

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

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.learning_rate)
