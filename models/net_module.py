"""Pytorch Lightning module for modeling."""


from utils.configuration import load_config
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import SGD, Adam
from pytorch_lightning.metrics.functional import accuracy


class NetModule(pl.LightningModule):

    def __init__(self, base_model, optimizer, learning_rate, validate_on_test=False):
        super().__init__()
        self.base_model = base_model
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = float(learning_rate)
        self.optimizer = Adam if optimizer == 'adam' else SGD
        self.validate_on_test = validate_on_test

    def forward(self, input):
        return self.base_model(input)

    def training_step(self, batch, batch_idx):
        input, labels = batch
        preds = self.forward(input)
        loss = self.loss(preds, labels)
        acc = accuracy(preds, labels)

        result = pl.TrainResult(loss)
        result.log_dict({
            'train_loss': loss,
            'train_acc': acc
        })

        return result

    def validation_step(self, batch, batch_idx):
        input, labels = batch
        preds = self.forward(input)
        loss = self.loss(preds, labels)
        acc = accuracy(preds, labels, num_classes=self.base_model.num_classes)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({
            'valid_loss': loss,
            'valid_acc': acc
        })

        return result

    def test_step(self, batch, batch_idx):
        if not self.validate_on_test:
            input, labels = batch
            preds = self.forward(input)
            loss = self.loss(preds, labels)
            acc = accuracy(preds, labels, num_classes=self.base_model.num_classes)

            result = pl.EvalResult(checkpoint_on=loss)
            result.log_dict({
                'test_loss': loss,
                'test_acc': acc
            })
        else:
            result = None

        return result

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.learning_rate)
