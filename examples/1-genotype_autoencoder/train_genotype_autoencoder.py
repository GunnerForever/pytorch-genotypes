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
    --weight-decay 1e-3 \
    --backend NumpyBackend \
    --backend-pickle-filename 1kg_common_norel.pkl

"""

import argparse
import functools

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


def _input_module(n_in, n_out):
    obj = nn.Sequential(
        ChunkDropout(n_in, 0.01, 100, 100),  # Hard coded for now...
        nn.Linear(n_in, n_out),
    )
    obj.out_features = n_out
    return obj


def train_genotype_autoencoder(args, _backend=None):
    # Parse encoder arguments and initialize.
    enc_layers = []
    enc_layers.append(
        ChunkPartiallyConnected(
            args.n_variants,
            args.enc_hidden[0],
            chunk_size=args.chunk_size,
            module_f=functools.partial(_input_module, n_out=args.enc_hidden[0])
        )
    )

    hidden_layer_sizes = args.enc_hidden[1:]
    output_layer_size = hidden_layer_sizes.pop()

    enc_layers.extend(build_mlp(
        enc_layers[0].get_effective_output_size(),
        hidden_layer_sizes,
        output_layer_size,
        add_batchnorm=args.enc_add_batchnorm,
        activations=[getattr(nn, s)() for s in args.enc_activations]
    ))

    encoder = nn.Sequential(*enc_layers)

    # Decoder.
    assert args.enc_hidden[-1] == args.dec_hidden[0]

    dec_layers = []
    dec_layers.extend(build_mlp(
        args.dec_hidden[0],
        args.dec_hidden[1:],
        add_batchnorm=args.dec_add_batchnorm,
        activations=[getattr(nn, s)() for s in args.dec_activations]
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
        backend = BACKENDS[args.backend].load(args.backend_pickle_filename)
    else:
        backend = _backend

    # Make it a dataset.
    dataset = GeneticDataset(backend)

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
        DataLoader(dataset, batch_size=128, num_workers=1, shuffle=True)
    )

    trainer.save_checkpoint("_autoencoder.ckpt")


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

    # Add genotype backend.
    geno = parser.add_argument_group(title="Genotype parameters.")
    geno.add_argument("--backend", choices=BACKENDS.keys(), required=True)
    geno.add_argument("--backend-pickle-filename", type=str, required=True)

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
