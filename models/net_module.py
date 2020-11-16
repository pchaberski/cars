"""Pytorch Lightning module for modeling."""


import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.metrics.functional import accuracy


class NetModule(pl.LightningModule):

    def __init__(
        self,
        base_model,
        loss,
        optimizer, optimizer_params={'lr': 0.001},
        lr_scheduler=None, lr_scheduler_params=None
    ):
        super().__init__()
        self.base_model = base_model
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params

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

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log_dict({
            'valid_loss': loss,
            'valid_acc': acc
        }, on_step=False, on_epoch=True)

        return result

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)

        if self.lr_scheduler is None:
            return [optimizer]
        else:
            scheduler = {
                'scheduler': self.lr_scheduler(optimizer, **self.lr_scheduler_params),
                'monitor': 'val_checkpoint_on',
                'name': 'lr'
            }
            return [optimizer], [scheduler]
