from typing import Iterable

import torch
from torch import nn

from ..models import GenotypeAutoencoder2Latent, MLP
from ..models.utils import build_mlp


class ChildBlockAutoencoder(GenotypeAutoencoder2Latent):
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
            lr=lr,
            weight_decay=weight_decay,
            use_standardized_genotype=use_standardized_genotype,
        )
        self.save_hyperparameters()

        if partial_chunk_size is not None:
            enc_kwargs = {
                "do_chunk": "input",
                "chunk_size": partial_chunk_size,
                "chunk_input_hidden_size": partial_connection_h
            }
            dec_kwargs = {
                "chunk_size": partial_chunk_size
            }
        else:
            enc_kwargs = dec_kwargs = {}

        self.encoder = MLP(
            chunk_size,
            enc_layers,
            rep_size,
            loss=None,
            lr=lr,
            add_hidden_layer_batchnorm=add_batchnorm,
            add_input_layer_batchnorm=not use_standardized_genotype,
            input_dropout_p=input_dropout_p,
            hidden_dropout_p=enc_h_dropout_p,
            activations=[getattr(nn, activation)()],
            **enc_kwargs
        )

        self.decoder = MLP(
            rep_size,
            dec_layers,
            chunk_size * 2,  # We'll use the parametetrization that uses two
            loss=None,       # latent variables for reconstruction.
            lr=lr,
            add_hidden_layer_batchnorm=add_batchnorm,
            add_input_layer_batchnorm=False,
            input_dropout_p=None,  # No dropout at repr. level.
            hidden_dropout_p=dec_h_dropout_p,
            activations=[getattr(nn, activation)()],
            **dec_kwargs
        )


class ParentBlockAutoencoder(GenotypeAutoencoder2Latent):
    def __init__(
        self,
        model1: MLP,
        model2: MLP,
        output_size: int,
        enc_layers: Iterable[int],
        dec_layers: Iterable[int],
        rep_size: int,
        lr: float,
        batch_size: int,
        max_epochs: int,
        weight_decay: float,
        add_batchnorm: bool,
        input_dropout_p: float,
        enc_h_dropout_p: float,
        dec_h_dropout_p: float,
        activation: str,
        use_standardized_genotype: bool,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            use_standardized_genotype=use_standardized_genotype,
            _save_hyperparameters=False
        )
        self.save_hyperparameters(ignore=["model1", "model2"])

        self.model1 = model1
        self.model2 = model2

        input_size = (
            self.model1.hparams.output_size +  # type: ignore
            self.model2.hparams.output_size    # type: ignore
        )

        enc_layer_mods = build_mlp(
            input_size,
            tuple(enc_layers),
            rep_size,
            add_batchnorm=add_batchnorm,
            activations=[getattr(nn, activation)()],
        )

        self.encoder = nn.Sequential(*enc_layer_mods)

        dec_layer_mods = build_mlp(
            rep_size,
            tuple(dec_layers),
            2 * output_size,
            add_batchnorm=add_batchnorm,
            activations=[getattr(nn, activation)()],
        )

        self.decoder = nn.Sequential(*dec_layer_mods)

        print(self)

    def encode(self, xs):
        cutpoint = self.model1.hparams.input_size
        x1 = xs[:, :cutpoint]
        x2 = xs[:, cutpoint:]

        z1 = self.model1(x1)
        z2 = self.model2(x2)

        z12 = torch.hstack((z1, z2))
        return self.encoder(z12)
