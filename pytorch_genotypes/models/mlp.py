"""
Convenience pytorch lightning multi-layer perceptron implementation.

This is provided mainly to help build more complex models.

"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import build_mlp


class MLP(pl.LightningModule):
    def __init__(self, lr, weight_decay, layers, activations, add_batchnorm,
                 loss):
        super().__init__()
        self.save_hyperparameters()

        if loss == "mse_loss":
            self.loss = F.mse_loss
        elif loss == "binary_cross_entropy_with_logits":
            self.loss = F.binary_cross_entropy_with_logits
        else:
            raise ValueError(f"'{loss}' is not currently supported by MLP.")

        if len(layers) < 3:
            raise ValueError("MLP needs at least one hidden layer.")

        self.input_size = layers[0]
        self.output_size = layers[-1]

        self.model = nn.Sequential(*build_mlp(
            layers[0],
            hidden=layers[1:-1],
            out=layers[-1],
            activations=[getattr(nn, s)() for s in activations],
            add_batchnorm=add_batchnorm,
        ))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return self.loss(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

    @staticmethod
    def add_model_specific_args(parent_parser, prefix="--"):
        parent_parser.add_argument(
            f"{prefix}layers",
            nargs="+",
            type=int,
            required=True
        )

        parent_parser.add_argument(
            f"{prefix}activations",
            nargs="+",
            type=str,
            required=True
        )

        parent_parser.add_argument(
            f"{prefix}add-batchnorm",
            action="store_true",
        )

        parent_parser.add_argument(
            f"{prefix}loss",
            choices=["mse_loss", "binary_cross_entropy_with_logits"],
            default="mse_loss"
        )

        return parent_parser
