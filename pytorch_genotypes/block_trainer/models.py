import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

from ..models import GenotypeAutoencoder, MLP


class ChildBlockAutoencoder(GenotypeAutoencoder):
    def __init__(
        self,
        chunk_size,
        enc_layers,
        dec_layers,
        rep_size,
        lr,
        batch_size,
        max_epochs,
        weight_decay,
        add_batchnorm,
        input_dropout_p,
        enc_h_dropout_p,
        dec_h_dropout_p,
        activation,
        use_standardized_genotype,
        partial_chunk_size,
        partial_connection_h,
    ):
        super().__init__(
            use_standardized_genotype=use_standardized_genotype,
        )
        self.save_hyperparameters()
        self.encoder = MLP(
            chunk_size,
            enc_layers,
            rep_size,
            loss=None,
            add_hidden_layer_batchnorm=add_batchnorm,
            add_input_layer_batchnorm=not use_standardized_genotype,
            input_dropout_p=input_dropout_p,
            hidden_dropout_p=enc_h_dropout_p,
            activations=[getattr(nn, activation)()],
            chunk_size=partial_chunk_size,
            do_chunk="input",
            chunk_input_hidden_size=partial_connection_h
        )

        self.decoder = MLP(
            rep_size,
            dec_layers,
            chunk_size * 2,
            loss=None,
            add_hidden_layer_batchnorm=add_batchnorm,
            add_input_layer_batchnorm=False,
            input_dropout_p=None,  # No dropout at repr. level.
            hidden_dropout_p=dec_h_dropout_p,
            activations=[getattr(nn, activation)()],
            chunk_size=partial_chunk_size,
        )


class ReshapeLogSoftmax(nn.Module):
    def __init__(self, n_snps):
        super().__init__()
        self.n_snps = n_snps

    def forward(self, x):
        x = x.view(-1, 3, self.n_snps)
        return F.log_softmax(x, dim=1)


class ParentModel(pl.LightningModule):
    def __init__(self, modelA, modelB, modelC, lr=1e-3, reg=1e-5):
        super(ParentModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

        self.lr = lr
        self.regularization = reg

    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.modelC(x)
        return x

    def training_step(self, batch, batch_idx):
        x1, x2 = batch
        z1 = self.modelA(x1)
        z2 = self.modelB(x2)

        z = torch.cat((z1, z2), dim=1)
        x_hat = self.modelC(z)

        x12 = torch.concat((x1, x2), dim=1)

        loss = F.nll_loss(x_hat, torch.round(x12).to(int))

        accuracy = (x12 == x_hat.argmax(dim=1)).to(float).mean(dim=0).mean()

        # Logging to TensorBoard by default
        self.log(f"Batch: {batch_idx} Train_loss", loss, on_epoch=True)
        self.log(f"Batch: {batch_idx} Accuracy", accuracy, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.regularization
        )
