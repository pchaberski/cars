"""Pytorch Lightning Module for Stanford Cars Dataset."""


import pytorch_lightning as pl


class StanfordLightningModule(pl.LightningModule):

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model.forward(x)
