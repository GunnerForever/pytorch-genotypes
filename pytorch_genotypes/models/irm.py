"""Implementation of the IRM algorithm.

This implementation is inspired by the DomainBed implementation available here:

https://github.com/facebookresearch/DomainBed

"""


import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torch.autograd as autograd
from torch.utils.data import TensorDataset

from ..dataset.utils import MultiEnvironmentIteratorFromSingleDataset


class LinearIRM(pl.LightningModule):
    def __init__(self, n_input, lr, irm_lam=1, loss="mse", weight_decay=0):
        super().__init__()
        self.save_hyperparameters()
        self.betas = nn.Linear(n_input, 1, dtype=torch.float32)

        if loss == "mse":
            self.loss = F.mse_loss
        elif loss == "bce":
            self.loss = F.binary_cross_entropy_with_logits
        else:
            raise ValueError(f"Unsupported loss '{loss}'.")

        print("Initializing IRM with hyperparameters:")
        print(self.hparams)

    def _irm_penalty(self, y_hat, y):
        w = torch.tensor(
            1.,
            dtype=torch.float32,
            device=self.device
        ).requires_grad_()

        loss_1 = self.loss(y_hat[::2] * w, y[::2])
        loss_2 = self.loss(y_hat[1::2] * w, y[1::2])

        grad_1 = autograd.grad(loss_1, [w], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [w], create_graph=True)[0]

        return torch.sum(grad_1 * grad_2)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train_")

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        return self._step(batch, batch_idx, "val_")

    def _step(self, batch, batch_idx, prefix=""):
        # env_ns = torch.tensor(
        #     [x_e.shape[0] for x_e, _ in batch],
        #     dtype=torch.float32,
        #     device=self.device
        # )
        # env_weights = env_ns / torch.sum(env_ns)
        # env_weights_iter = iter(env_weights)

        loss = 0.0
        penalty = 0.0

        for x_e, y_e in batch:
            x_e = x_e.to(torch.float32)
            y_e = y_e.to(torch.float32)

            y_hat_e = self.betas(x_e)
            env_loss = self.loss(y_hat_e, y_e)
            env_penalty = self._irm_penalty(y_hat_e, y_e)

            loss += env_loss
            penalty += env_penalty

        loss /= len(batch)
        penalty /= len(batch)

        l2 = torch.norm(self.betas.weight, 2)
        irm_loss = (
            loss +
            (self.hparams.irm_lam * penalty) +
            (self.hparams.weight_decay * l2)
        )

        self.log(f"{prefix}penalty", penalty)
        self.log(f"{prefix}erm_loss", loss)
        self.log(f"{prefix}irm_loss", irm_loss)
        self.log("l2", l2)

        return irm_loss

    def configure_optimizers(self):
        return torch.optim.LBFGS(
            self.parameters(),
            lr=self.hparams.lr,
        )


def test():
    model = LinearIRM(2, lr=1)
    trainer = pl.Trainer()
    trainer.fit(model, example_dataset())


def example_dataset():
    env1 = torch.hstack((
        torch.arange(10).view(-1, 1),
        torch.ones(10).view(-1, 1)
    ))
    env2 = torch.hstack((
        torch.arange(3).view(-1, 1),
        torch.ones(3).view(-1, 1) + 1
    ))

    mat = torch.vstack((env1, env2))
    dataset = TensorDataset(mat)

    env_indices = {
        1: list(range(10)),
        2: list(range(11, 13))
    }

    iterator = MultiEnvironmentIteratorFromSingleDataset(
        dataset,
        env_indices,
        batch_size=2
    )
    return iterator


if __name__ == "__main__":
    test()
