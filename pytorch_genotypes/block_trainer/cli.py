"""
Command-line interface entrypoint for the block trainer utility tool.

"""

import os
import argparse
import pprint
from typing import Dict
from pkg_resources import resource_filename

from pytorch_genotypes.dataset.core import FixedSizeChunks

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .models import ChildBlockAutoencoder

from ..dataset import BACKENDS


try:
    from orion.client import report_objective
    ORION_SWEEP = True
except ImportError:
    ORION_SWEEP = False


DEFAULT_TEMPLATE = resource_filename(
    __name__, os.path.join("config_files", "block_trainer_bash.sh")
)


DEFAULT_CONFIG = resource_filename(
    __name__, os.path.join("config_files", "block_trainer_model_config.py")
)


def parse_config(filename):
    conf_global = {}
    with open(filename) as f:
        script = f.read()
        exec(script, conf_global)

    conf = conf_global["get_block_trainer_config"]()
    assert isinstance(conf, dict)
    return conf


def train(args):
    config = parse_config(args.config)

    print("Current model configuration:")
    pprint.pprint(config)

    test_proportion = 0.1

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
        batch_size=config["batch_size"],
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=n_test,
        num_workers=1
    )

    model = ChildBlockAutoencoder(
        chunk_size=args.chunk_size,
        enc_layers=config["model/enc_layers"],
        dec_layers=config["model/dec_layers"],
        rep_size=config["model/rep_size"],
        lr=config["lr"],
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
        weight_decay=config["weight_decay"],
        add_batchnorm=config["add_batchnorm"],
        input_dropout_p=config["input_dropout_p"],
        enc_h_dropout_p=config["enc_h_dropout_p"],
        dec_h_dropout_p=config["dec_h_dropout_p"],
        activation=config["model/activation"]
    )

    logger = None
    if args.wandb_logger:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(
            project=f"block_trainer_block_{args.chunk_index}"
        )
        logger = wandb_logger

    stop_if_nan = EarlyStopping(
        monitor="reconstruction_nll",
        check_finite=True
    )

    trainer = pl.Trainer(
        log_every_n_steps=1,
        accelerator="gpu",
        devices=-1,  # use all GPUs
        max_epochs=config["max_epochs"],
        logger=logger,
        callbacks=[stop_if_nan]
    )
    trainer.fit(model, train_loader)
    test_results = trainer.test(model, test_loader)[0]

    if ORION_SWEEP:
        report_objective(test_results["test_reconstruction_nll"])

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
    exec_parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Python configuration file for hyperparameters and autoencoder "
             f"model. See {DEFAULT_CONFIG} for an example."
    )
    exec_parser.add_argument("--wandb-logger", action="store_true")

    return parser.parse_args()
