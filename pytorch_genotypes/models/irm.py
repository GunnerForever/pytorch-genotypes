"""Implementation of the IRM algorithm.

This implementation is inspired by the DomainBed implementation available here:

https://github.com/facebookresearch/DomainBed

"""

from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torch.autograd as autograd
from torch.utils.data import TensorDataset

from ..dataset.utils import MultiEnvironmentIteratorFromSingleDataset


class LinearIRM(pl.LightningModule):
    def __init__(self, n_input, lr, irm_lam=1, loss="mse", weight_decay=0,
                 l1_penalty=0):
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
        loss = 0.0
        penalty = 0.0

        # Calculate weight norms for penalized models.
        l1 = torch.norm(self.betas.weight, 1)
        l2 = torch.norm(self.betas.weight, 2)

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

        irm_loss = (
            loss +
            (self.hparams.irm_lam * penalty) +
            (self.hparams.weight_decay * l2) +
            (self.hparams.l1_penalty * l1)
        )

        self.log(f"{prefix}penalty", penalty)
        self.log(f"{prefix}erm_loss", loss)
        self.log(f"{prefix}irm_loss", irm_loss)
        self.log("l2", l2)
        self.log("l1", l1)

        return irm_loss

    def configure_optimizers(self):
        return torch.optim.LBFGS(
            self.parameters(),
            lr=self.hparams.lr,
        )


class WindowedLinearIRM(LinearIRM):
    def __init__(
        self,
        windows: torch.Tensor,
        *args, **kwargs
    ):
        """Linear IRM where the penalty is applied in windows.

        The windows should be defined as a p by 2 tensor where p is the number
        of windows.

        """
        super().__init__(*args, **kwargs)
        self._penalty_history = []

    def get_penalties(self, x_e, y_e):
        n_windows = self.hparams.windows.shape[0]

        # Compute penalty by window.
        penalties = torch.empty(
            n_windows,
            dtype=torch.float32,
            device=self.device
        )

        for i in range(n_windows):
            left, right = self.hparams.windows[i, :]

            x_e_local = x_e[:, left:(right+1)]  # n x m
            w = self.betas.weight[:, left:(right+1)]  # 1 x m
            y_hat_e_local = x_e_local @ w.T + self.betas.bias

            window_penalty = self._irm_penalty(y_hat_e_local, y_e)
            penalties[i] = window_penalty

        return penalties

    def on_train_end(self):
        import json
        with open("_penalty_history.json", "wt") as f:
            json.dump(self._penalty_history, f)

    def _step(self, batch, batch_idx, prefix=""):
        loss = 0.0
        penalties: Optional[torch.Tensor] = None

        # Calculate weight norms for penalized models.
        l1 = torch.norm(self.betas.weight, 1)
        l2 = torch.norm(self.betas.weight, 2)

        for x_e, y_e in batch:
            x_e = x_e.to(torch.float32)
            y_e = y_e.to(torch.float32)

            y_hat_e = self.betas(x_e)
            env_loss = self.loss(y_hat_e, y_e)

            penalties_e = self.get_penalties(x_e, y_e)

            loss += env_loss
            if penalties is None:
                penalties = penalties_e
            else:
                penalties += penalties_e

        loss /= len(batch)
        penalty = torch.norm(penalties) / len(batch)

        self._penalty_history.append(
            (self.global_step, penalties.detach().numpy().tolist())
        )

        irm_loss = (
            loss +
            (self.hparams.irm_lam * penalty) +
            (self.hparams.weight_decay * l2) +
            (self.hparams.l1_penalty * l1)
        )

        self.log(f"{prefix}penalty", penalty)
        self.log(f"{prefix}erm_loss", loss)
        self.log(f"{prefix}irm_loss", irm_loss)
        self.log("l2", l2)
        self.log("l1", l1)

        return irm_loss


def make_fixed_windows(seq_len: int, window_size: int) -> torch.Tensor:
    """Make fixed-size windows for the WindowedLinearIRM."""
    windows = []
    left = 0
    right = 0

    while left <= seq_len - 1:
        right = left + window_size - 1
        if right >= seq_len:
            right = seq_len - 1

        windows.append((left, right))
        left = right + 1

    return torch.tensor(windows)


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
