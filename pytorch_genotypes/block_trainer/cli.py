"""
Command-line interface entrypoint for the block trainer utility tool.

"""

import os
import argparse
from typing import Dict
from pkg_resources import resource_filename

from pytorch_genotypes.dataset.core import FixedSizeChunks

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import pytorch_lightning as pl

from pytorch_genotypes.modules.chunk_dropout import ChunkDropout

from ..dataset import BACKENDS
from ..models import GenotypeProbabilisticAutoencoder
from ..models.utils import build_mlp


DEFAULT_TEMPLATE = resource_filename(
    __name__, os.path.join("command_templates", "block_trainer_bash.sh")
)


REP_SIZE = 256


def build_encoder(args):
    # 1k, 128, 256 <> 256, 512, 1k (lr=1e-3, bs=512, epochs=1000)
    enc_layers = [
        nn.BatchNorm1d(args.chunk_size),
    ]

    enc_layers.extend(build_mlp(
        args.chunk_size, (712, ), REP_SIZE,
        activations=[nn.LeakyReLU()],
        add_batchnorm=True
    ))
    # ChunkDropout(args.chunk_size, 0.1, 1, 20, weight_scaling=True)
    # Try long but few dropouts (effective ~15.4%). (BAD)
    # ChunkDropout(args.chunk_size, 0.001, 300, 100, weight_scaling=True)
    return nn.Sequential(*enc_layers)


def build_decoder(args):
    return nn.Linear(REP_SIZE, args.chunk_size)
    # return nn.Sequential(*build_mlp(
    #     REP_SIZE, (128, ), args.chunk_size,
    #     activations=[nn.LeakyReLU()],
    #     add_batchnorm=True
    # ))


def train(args):
    lr = 1e-3
    batch_size = 256
    n_epochs = 500
    test_proportion = 0.1
    weight_decay = 0  # 1e-5

    print("-----------------------------------")
    print("Block Training Process Begin.")
    print("-----------------------------------")
    backend = BACKENDS[args.backend].load(args.backend_pickle_filename)
    chunks = FixedSizeChunks(backend, max_variants_per_chunk=args.chunk_size)

    genotypes_dataset = chunks.get_dataset_for_chunk_id(args.chunk_index)

    n = backend.get_n_samples()

    # Sample some test indices.
    n_test = int(round(test_proportion * n))
    n_train = n - n_test

    train_dataset, test_dataset = random_split(
        genotypes_dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=n_train if batch_size == "full" else batch_size,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=n_test,
        num_workers=1
    )

    model = GenotypeProbabilisticAutoencoder(
        encoder=build_encoder(args),
        decoder=build_decoder(args),
        lr=lr,
        weight_decay=weight_decay
    )

    trainer = pl.Trainer(
        log_every_n_steps=1,
        gpus=-1,  # use all GPUs
        max_epochs=n_epochs,
    )
    trainer.fit(model, train_loader)
    trainer.test(model, test_loader)
    print("-----------------------------------")
    print("Training process has finished. Saving trained model.")
    print("-----------------------------------")

    # Save model
    results_base = "chunk_checkpoints"
    if not os.path.isdir(results_base):
        os.makedirs(results_base)

    checkpoint_filename = os.path.join(
        results_base, f"model-block-{args.chunk_index}.ckpt"
    )
    trainer.save_checkpoint(checkpoint_filename)
    print("-----------------------------------")
    print("Model Saved.")
    print("-----------------------------------")


def main():
    args = parse_args()
    backend = BACKENDS[args.backend].load(args.backend_pickle_filename)
    chunks = FixedSizeChunks(backend, max_variants_per_chunk=args.chunk_size)

    if args.mode == "train":
        return train(args)

    if args.template is None:
        template = DEFAULT_TEMPLATE
    else:
        template = args.template

    with open(template, "rt") as f:
        template = f.read()

    template_params = parse_template_params(args.template_params)

    command = (
        f"pt-geno-block-trainer "
        f"--backend {args.backend} "
        f'--backend-pickle-filename "{args.backend_pickle_filename}" '
        f"--chunk-size {args.chunk_size} "
    )

    script = template.format(
        command=command, n_blocks=len(chunks), **template_params
    )

    with open(args.output_script, "w") as f:
        f.write(script)

    # TODO
    print(
        f"<<< BLOCK TRAINER >>>\n\n"
        f"Block training is done in two steps.\n\n"
        f"First, run the '{args.output_script}' file that was generated.\n\n"
        f"By default, this is a bash script that will train all of the first "
        f"level blocks in parallel using as many CPUs as specified and using "
        f"xargs for parallelism. You may edit the file as needed to suit your "
        f"computing environment.\n\n"
        f"Once this is done, run the step 2 (to be documented)."
    )


def parse_template_params(params: str) -> Dict[str, str]:
    """Parsers template arguments in the format key1=value1,key2=value2,..."""
    if not params:
        return {}

    parameters = {}
    for arg_pair in params.split(","):
        k, v = arg_pair.split("=")
        parameters[k] = v

    return parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Backend configuration.
    parser.add_argument("--backend", choices=BACKENDS.keys(), required=True)
    parser.add_argument(
        "--backend-pickle-filename",
        help="Path to a file containing a serialized backend that is an "
        "instance of the class specified by '--backend'.",
        type=str,
        required=True,
    )

    # Chunking configuration.
    # For now, we only allow fixed size chunks.
    parser.add_argument(
        "--chunk-size",
        default=1000,
        type=int,
        help="Number of contiguous variants that are jointly modeled in the "
        "blocks of the first level.",
    )

    # Script template.
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("--template-params", type=str, default="n_cpus=1")

    # Output script.
    parser.add_argument(
        "--output-script",
        type=str,
        default="run_step1_block_trainer_all_blocks.sh",
    )

    # Subparser.
    subparsers = parser.add_subparsers(dest="mode")
    exec_parser = subparsers.add_parser("train")
    exec_parser.add_argument("--chunk-index", required=True, type=int)

    return parser.parse_args()
