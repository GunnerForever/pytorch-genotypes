import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F


class ChildModel(pl.LightningModule):
    def __init__(self, input_dim, n_hidden, lr=1e-3, drop=0.25, reg=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Dropout(self.hparams.drop),
            nn.Linear(input_dim, n_hidden),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_hidden, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 3 * input_dim),
            ReshapeLogSoftmax(n_snps=input_dim),
        )

    def forward(self, features):
        reconstruction = self.encoder(features)
        reconstruction = self.decoder(reconstruction)
        return reconstruction

    def training_step(self, batch, batch_idx):
        # Training_step defined the train loop.
        # It is independent of forward
        x = batch[0]
        x = x.to(torch.float32)

        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.nll_loss(x_hat, torch.round(x).to(int))
        # Logging to TensorBoard by default
        self.log(f"Batch: {batch_idx} Train Loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.reg
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
