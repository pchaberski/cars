"""Pytorch Lightning module for modeling."""


from utils.configuration import load_config
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import SGD, Adam
from pytorch_lightning.metrics.functional import accuracy


class NetModule(pl.LightningModule):

    def __init__(
        self, base_model, optimizer, learning_rate, 
        lr_scheduler=None, lr_scheduler_params=None,
        validate_on_test=False
    ):
        super().__init__()
        self.base_model = base_model
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = float(learning_rate)
        self.optimizer = Adam if optimizer == 'adam' else SGD
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.validate_on_test = validate_on_test

    def forward(self, input):
        return self.base_model(input)

    def training_step(self, batch, batch_idx):
        input, labels = batch
        preds = self.forward(input)
        pred_classes = torch.argmax(preds, dim=1)

        loss = self.loss(preds, labels)
        acc = accuracy(pred_classes, labels, num_classes=self.base_model.num_classes)

        result = pl.TrainResult(loss)
        result.log_dict({
            'train_loss': loss,
            'train_acc': acc
        }, on_step=False, on_epoch=True)

        return result

    def validation_step(self, batch, batch_idx):
        input, labels = batch
        preds = self.forward(input)
        pred_classes = torch.argmax(preds, dim=1)

        loss = self.loss(preds, labels)
        acc = accuracy(pred_classes, labels, num_classes=self.base_model.num_classes)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({
            'valid_loss': loss,
            'valid_acc': acc
        }, on_step=False, on_epoch=True)

        return result

    def test_step(self, batch, batch_idx):
        if not self.validate_on_test:
            input, labels = batch
            preds = self.forward(input)
            pred_classes = torch.argmax(preds, dim=1)

            loss = self.loss(preds, labels)
            acc = accuracy(pred_classes, labels, num_classes=self.base_model.num_classes)

            result = pl.EvalResult(checkpoint_on=loss)
            result.log_dict({
                'test_loss': loss,
                'test_acc': acc
            }, on_step=False, on_epoch=True)
        else:
            result = None

        return result

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)

        if self.lr_scheduler is None:
            return optimizer
        else:
            scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_params)
            return [optimizer], [scheduler]
