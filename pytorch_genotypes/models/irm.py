"""Implementation of the IRM algorithm.

This implementation is inspired by the DomainBed implementation available here:

https://github.com/facebookresearch/DomainBed

"""

import json
from typing import Optional, List

import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import torch.autograd as autograd
from torch.utils.data import TensorDataset

from ..dataset.utils import MultiEnvironmentIteratorFromSingleDataset


class LinearIRM(pl.LightningModule):
    def __init__(self, n_input, lr, irm_lam=1, loss="mse", weight_decay=0,
                 l1_penalty=0, intercept=True):
        super().__init__()
        self.save_hyperparameters()
        self.betas = nn.Linear(
            n_input, 1, dtype=torch.float32,
            bias=bool(intercept)
        )

        if loss == "mse":
            self.loss = F.mse_loss
        elif loss == "bce":
            self.loss = F.binary_cross_entropy_with_logits
        else:
            raise ValueError(f"Unsupported loss '{loss}'.")

        print("Initializing IRM with hyperparameters:")
        print(self.hparams)

    @staticmethod
    def _loss_and_irm_penalty_static(
        x,
        y,
        betas,
        intercept=0,
        loss=F.mse_loss,
        device=None
    ):
        # betas: 1 x p
        # x: mb x p
        # y: mb x 1
        w = torch.tensor(1., device=device).requires_grad_()

        if not intercept:
            intercept = 0

        scaled_yhat = w * (x @ betas.T) + intercept

        full_loss = loss(scaled_yhat, y, reduction="none")

        loss_1 = torch.mean(full_loss[::2])
        loss_2 = torch.mean(full_loss[1::2])

        grad_1 = autograd.grad(loss_1, [w], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [w], create_graph=True)[0]

        return torch.mean(full_loss), torch.sum(grad_1 * grad_2)

    def _loss_and_irm_penalty(self, x, y):
        return self._loss_and_irm_penalty_static(
            x=x,
            y=y,
            betas=self.betas.weight,
            intercept=self.betas.bias,
            loss=self.loss,
        )

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

            env_loss, env_penalty = self._loss_and_irm_penalty(x_e, y_e)

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
        *args,
        penalty_history: bool = False,
        **kwargs
    ):
        """Linear IRM where the penalty is applied in windows.

        The windows should be defined as a p by 2 tensor where p is the number
        of windows.

        """
        super().__init__(*args, **kwargs)

        if penalty_history:
            self._penalty_histories: Optional[List[torch.Tensor]] = []
        else:
            self._penalty_histories = None

    def get_prediction_and_local_penalties(self, x_e, y_e):
        n_windows = self.hparams.windows.shape[0]

        # Compute penalty by window.
        penalties = torch.empty(
            n_windows,
            dtype=torch.float32,
            device=self.device
        )

        y_hat = self.betas.bias
        if y_hat is None:
            y_hat = 0

        for i in range(n_windows):
            left, right = self.hparams.windows[i, :]

            x_e_local = x_e[:, left:(right+1)]  # n x m
            window_betas = self.betas.weight[:, left:(right+1)]  # 1 x m

            # FIXME
            raise RuntimeError()
            y_e_local = y_e - torch.mean(y_e[x_e_local == 0])

            _, window_penalty = self._loss_and_irm_penalty_static(
                x_e_local,
                y_e_local,  # FIXME
                betas=window_betas,
                intercept=self.betas.bias,
                loss=self.loss,
                device=self.device
            )

            penalties[i] = window_penalty
            y_hat = y_hat + x_e_local @ window_betas.T

        return y_hat, penalties

    def on_train_end(self):
        with open("_penalty_history.json", "wt") as f:
            json.dump([penalties.detach().numpy().tolist()
                       for penalties in self._penalty_histories], f)

    def _step(self, batch, batch_idx, prefix=""):
        loss = 0.0
        penalty = 0.0

        n_windows = self.hparams.windows.shape[0]

        # Calculate weight norms for penalized models.
        l1 = torch.norm(self.betas.weight, 1)
        l2 = torch.norm(self.betas.weight, 2)

        track_penalties = self._penalty_histories is not None

        if track_penalties:
            cur_penalties = torch.zeros(n_windows)

        for x_e, y_e in batch:
            x_e = x_e.to(torch.float32)
            y_e = y_e.to(torch.float32)

            y_hat_e, penalties = self.get_prediction_and_local_penalties(
                x_e, y_e
            )

            if track_penalties:
                cur_penalties += penalties

            env_loss = self.loss(y_hat_e, y_e)

            loss += env_loss
            penalty += torch.norm(penalties) / n_windows

        if track_penalties:
            self._penalty_histories.append(cur_penalties / n_windows)

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
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )


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
