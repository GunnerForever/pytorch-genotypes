"""
Convenience pytorch lightning multi-layer perceptron implementation.

This is provided mainly to help build more complex models.

"""

from typing import List, Optional, Iterable, Callable, Dict
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import ChunkPartiallyConnected

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

from .utils import build_mlp


LossString = Literal[
    "mse",
    "binary_cross_entropy_with_logits",
    "cross_entropy"
]


LOSS_MAP: Dict[LossString, Callable] = {
    "mse": F.mse_loss,
    "binary_cross_entropy_with_logits": F.binary_cross_entropy_with_logits,
    "cross_entropy": F.cross_entropy,
}


class MLP(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_layers: Iterable[int],
        output_size: int,
        loss: Optional[LossString],
        lr: float,
        use_dosage: bool = True,
        do_chunk: Optional[Literal["input", "output"]] = None,
        chunk_size: Optional[int] = None,
        chunk_input_hidden_size: Optional[int] = None,
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

        if do_chunk is not None and chunk_size is None:
            raise ValueError("Provide a chunk_size if chunking is requested.")

        if do_chunk == "input":
            assert chunk_size is not None
            if chunk_input_hidden_size is None:
                raise ValueError("Input partial connection need a "
                                 "'chunk_input_hidden_size' parameter.")

            partial_layer = ChunkPartiallyConnected(
                input_size,
                chunk_input_hidden_size,
                chunk_size=chunk_size,
            )
            modules.append(partial_layer)
            input_size = partial_layer.get_effective_output_size()

        if hidden_dropout_p is not None and hidden_dropout_p > 0:
            activations.append(nn.Dropout(hidden_dropout_p))

        hidden_layers = tuple(hidden_layers)

        modules.extend(build_mlp(
            input_size,
            hidden=hidden_layers,
            out=output_size if do_chunk != "output" else None,
            activations=activations,
            add_batchnorm=add_hidden_layer_batchnorm,
        ))

        if do_chunk == "output":
            assert chunk_size is not None

            modules.append(
                ChunkPartiallyConnected(
                    hidden_layers[-1],
                    output_size,
                    chunk_size=chunk_size,
                    chunk_the_input=False
                )
            )

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
        loss = self.loss(y_hat, batch.endogenous)

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        if self.hparams.weight_decay is not None:
            wd = self.hparams.weight_decay
        else:
            wd = 0

        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                weight_decay=wd)
