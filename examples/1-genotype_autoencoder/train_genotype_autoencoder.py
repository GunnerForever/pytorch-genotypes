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
from torch.utils.data import DataLoader, random_split

from pytorch_genotypes.models import GenotypeProbabilisticAutoencoder
from pytorch_genotypes.models.utils import build_mlp
from pytorch_genotypes.modules import (
    ChunkPartiallyConnected,
    ChunkDropout
)
from pytorch_genotypes.dataset import BACKENDS, GeneticDataset


def train_genotype_autoencoder(args, _backend=None):
    print("Initializing weights")

    # Parse encoder arguments and initialize.
    enc_layers = [
        nn.BatchNorm1d(args.n_variants),
        ChunkPartiallyConnected(
            args.n_variants,
            args.enc_hidden[0],
            chunk_size=5000
        )
    ]

    hidden_layer_sizes = args.enc_hidden[1:]
    output_layer_size = hidden_layer_sizes.pop()

    enc_layers.extend(build_mlp(
        enc_layers[-1].get_effective_output_size(),
        hidden_layer_sizes,
        output_layer_size,
        add_batchnorm=args.enc_add_batchnorm,
    ))

    enc_layers = nn.Sequential(*enc_layers)

    encoder = nn.Sequential(*enc_layers)
    decoder = nn.Linear(7, args.n_variants, bias=False)

    print("Initializing backend")

    # Initialize backend.
    if _backend is None:
        backend = BACKENDS["NumpyBackend"].load(args.backend)
    else:
        backend = _backend

    print("Making datasets")

    # Make it a dataset.
    if args.test_proportion > 0:
        n_test = round(len(backend) * args.test_proportion)
        n_train = len(backend) - n_test

        train_dataset, test_dataset = random_split(
            GeneticDataset(backend, genotype_standardization=False),
            [n_train, n_test]
        )

    else:
        train_dataset = GeneticDataset(backend, genotype_standardization=False)
        test_dataset = None

    print("Initialize final model and start training.")

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

    print("Starting fit.")
    trainer.fit(
        autoencoder,
        DataLoader(train_dataset, batch_size=65)
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
