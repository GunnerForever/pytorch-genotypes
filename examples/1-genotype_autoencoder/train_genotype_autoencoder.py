#!/usr/bin/env python

"""
Example on how to train a probabilistic autoencoder using pytorch-genotypes.

./train_genotype_autoencoder.py \
    --n-variants 1664852 \
    --chunk-size 5000 \
    --enc-hidden 10 128 32 \
    --enc-activations LeakyReLU \
    --dec-hidden 32 128 10 \
    --dec-activations LeakyReLU \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --backend 1kg_common_norel.pkl \
    --test-proportion 0.1

"""

import argparse

import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch_genotypes.models import GenotypeProbabilisticAutoencoder
from pytorch_genotypes.models.utils import build_mlp
from pytorch_genotypes.modules import (
    ChunkPartiallyConnected,
    ChunkDropout
)
from pytorch_genotypes.dataset import BACKENDS, GeneticDataset


class InputModuleFactory(object):
    """Class used to cache the effective dropout rate efficiently."""
    def __init__(self, n_out):
        self.n_out = n_out
        self.scaling_factor = None

    def _get_dropout_mod(self, n_in, *args, **kwargs):
        return ChunkDropout(n_in, 0.01, 100, 100, *args, **kwargs)

    def get_module(self, n_in):
        if self.scaling_factor is None:
            dropout = self._get_dropout_mod(n_in)
            self.scaling_factor = dropout.get_scaling_factor()
        else:
            dropout = self._get_dropout_mod(n_in, weight_scaling=False)
            dropout.set_scaling_factor(self.scaling_factor)

        obj = nn.Sequential(dropout, nn.Linear(n_in, self.n_out))
        obj.out_features = self.n_out

        return obj


def train_genotype_autoencoder(args, _backend=None):
    input_fac = InputModuleFactory(n_out=args.enc_hidden[0])

    # Parse encoder arguments and initialize.
    enc_layers = []
    enc_layers.append(
        ChunkPartiallyConnected(
            args.n_variants,
            args.enc_hidden[0],
            chunk_size=args.chunk_size,
            module_f=input_fac.get_module
        )
    )

    hidden_layer_sizes = args.enc_hidden[1:]
    output_layer_size = hidden_layer_sizes.pop()

    activations = []
    for s in args.enc_activations:
        activations.append(getattr(nn, s)())
        activations.append(nn.Dropout(0.5))  # TODO parametrize.

    enc_layers.extend(build_mlp(
        enc_layers[0].get_effective_output_size(),
        hidden_layer_sizes,
        output_layer_size,
        add_batchnorm=args.enc_add_batchnorm,
        activations=activations
    ))

    encoder = nn.Sequential(*enc_layers)

    # Decoder.
    assert args.enc_hidden[-1] == args.dec_hidden[0]

    activations = []
    for s in args.dec_activations:
        activations.append(getattr(nn, s)())
        activations.append(nn.Dropout(0.5))  # TODO parametrize.

    dec_layers = []
    dec_layers.extend(build_mlp(
        args.dec_hidden[0],
        args.dec_hidden[1:],
        add_batchnorm=args.dec_add_batchnorm,
        activations=activations
    ))

    dec_layers.append(ChunkPartiallyConnected(
        args.dec_hidden[-1],
        args.n_variants,
        chunk_size=args.chunk_size,
        chunk_the_input=False
    ))

    decoder = nn.Sequential(*dec_layers)

    # Initialize backend.
    if _backend is None:
        backend = BACKENDS["NumpyBackend"].load(args.backend)
    else:
        backend = _backend

    # Make it a dataset.
    if args.test_proportion > 0:
        n_test = round(len(backend) * args.test_proportion)
        test_backend, train_backend = backend.split_samples(n_test)

        train_dataset = GeneticDataset(train_backend)
        test_dataset = GeneticDataset(test_backend)

    else:
        train_dataset = GeneticDataset(backend)
        test_dataset = None

    # Initialize the autoencoder.
    autoencoder = GenotypeProbabilisticAutoencoder(
        lr=args.lr,
        weight_decay=args.weight_decay,
        encoder=encoder,
        decoder=decoder
    )

    print(autoencoder)

    # Train the model.
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(
        autoencoder,
        DataLoader(train_dataset, batch_size=128, num_workers=1, shuffle=True)
    )
    del train_dataset

    print("Saving checkpoint...")
    trainer.save_checkpoint("_autoencoder.ckpt")

    if test_dataset:
        trainer.test(
            autoencoder,
            DataLoader(test_dataset, batch_size=len(test_dataset) // 5)
        )


def parse_args():
    parser = argparse.ArgumentParser()

    # Add architecture parameters.
    parser.add_argument("--n-variants", required=True, type=int)
    parser.add_argument("--chunk-size", required=True, type=int, default=2000)

    enc = parser.add_argument_group(title="Encoder architecture.")
    enc.add_argument("--enc-hidden", nargs="+", required=True, type=int)
    enc.add_argument("--enc-activations", nargs="+", required=True, type=str)
    enc.add_argument("--enc-add-batchnorm", action="store_true")

    dec = parser.add_argument_group(title="Decoder architecture.")
    dec.add_argument("--dec-hidden", nargs="+", required=True, type=int)
    dec.add_argument("--dec-activations", nargs="+", required=True, type=str)
    dec.add_argument("--dec-add-batchnorm", action="store_true")

    # Add training hyperparameters.
    trainer = parser.add_argument_group(title="Trainer parameters.")
    trainer.add_argument("--lr", help="Learning rate", type=float,
                         required=True)
    trainer.add_argument("--weight-decay", help="Weight decay", type=float,
                         required=True)

    trainer.add_argument("--test-proportion", type=float, default=0.1,
                         help="Proportion of samples to use as test set.")

    # Add genotype backend.
    geno = parser.add_argument_group(title="Genotype parameters.")
    geno.add_argument(
        "--backend",
        type=str,
        required=True,
        help="Path to pickle containing NumpyBackend.\n"
             "For now, we only allow this backend to allow easy sample "
             "splitting which is hard to implement with the ZarrBackend for "
             "performance reasons."
    )

    # Add trainer parameters.
    # pl.Trainer.add_argparse_args(parser)
    trainer = parser.add_argument_group(title="pl.Trainer")
    trainer.add_argument("--accelerator", default=None)
    trainer.add_argument("--devices", default=None)
    trainer.add_argument("--log-every-n-steps", default=1, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_genotype_autoencoder(args)
