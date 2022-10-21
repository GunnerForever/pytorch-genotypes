"""
Convenience pytorch lightning multi-layer perceptron implementation.

This is provided mainly to help build more complex models.

"""

from typing import List, Optional, Iterable, Callable, Dict
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

from .utils import build_mlp


LossString = Literal["mse", "binary_cross_entropy_with_logits"]


LOSS_MAP: Dict[LossString, Callable] = {
    "mse": F.mse_loss,
    "binary_cross_entropy_with_logits": F.binary_cross_entropy_with_logits
}


class MLP(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_layers: Iterable[int],
        output_size: int,
        loss: Optional[LossString],
        use_dosage: bool = True,
        weight_decay: Optional[float] = None,
        add_hidden_layer_batchnorm: bool = False,
        add_input_layer_batchnorm: bool = True,
        input_dropout_p: Optional[float] = None,
        hidden_dropout_p: Optional[float] = None,
        activations: List[nn.Module] = [nn.ReLU()],
    ):
        super().__init__()
        self.save_hyperparameters()
        if loss is not None:
            self.loss: Optional[Callable] = LOSS_MAP[loss]
        else:
            self.loss = None

        modules: List[nn.Module] = []
        if add_input_layer_batchnorm:
            modules.append(nn.BatchNorm1d(input_size))

        if input_dropout_p is not None and input_dropout_p > 0:
            modules.append(nn.Dropout(input_dropout_p))

        if hidden_dropout_p is not None and hidden_dropout_p > 0:
            activations.append(nn.Dropout(hidden_dropout_p))

        modules.extend(build_mlp(
            input_size,
            hidden=tuple(hidden_layers),
            out=output_size,
            activations=activations,
            add_batchnorm=add_hidden_layer_batchnorm,
        ))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

    def _extract_input(self, batch):
        if len(batch) == 1:
            raise ValueError("MLP needs supervised ")

    def training_step(self, batch, batch_idx):
        if self.loss is None:
            raise RuntimeError(
                "Specify a loss at MLP initialization to enable training."
            )

        if not hasattr(batch, "endogenous"):
            raise ValueError(
                "No endogenous (Y) value for supervised learning."
            )

        if self.hparams.use_dosage:
            x = batch.dosage
        else:
            x = batch.std_genotypes

        if hasattr(batch, "exogenous"):
            # Include covariates.
            x = torch.hstack([x, batch.exogenous])

        y_hat = self.forward(x)

        return self.loss(y_hat, batch.endogenous)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
