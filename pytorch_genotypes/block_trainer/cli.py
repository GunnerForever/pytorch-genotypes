"""
Command-line interface entrypoint for the block trainer utility tool.

"""

import os
import argparse
import pprint
from typing import Dict
from pkg_resources import resource_filename

from pytorch_genotypes.dataset.core import FixedSizeChunks

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, Callback
)

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


VAL_INDICES_FILENAME = "block-trainer-val-indices.npz"

OBJECTIVE = "mean_like"


class OrionCallback(Callback):
    def __init__(self, monitor_metric: str):
        self.metric = monitor_metric
        self.best_metric = float("inf")

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        cur_metric = trainer.callback_metrics[self.metric].cpu().numpy().item()
        if cur_metric < self.best_metric:
            self.best_metric = cur_metric

    def on_fit_end(self, trainer, pl_module):
        report_objective(self.best_metric)


class StopIfNan(EarlyStopping):
    def __init__(self):
        super().__init__(
            monitor=f"train_{OBJECTIVE}",
            check_finite=True,
        )

    def _run_early_stopping_check(self, trainer):
        logs = trainer.callback_metrics
        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)
        if should_stop:
            print("\nStopping training because training metric is NaN.")
            super()._run_early_stopping_check(trainer)


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

    print("-----------------------------------")
    print("Block Training Process Begin.")
    print("-----------------------------------")
    backend = BACKENDS[args.backend].load(args.backend_pickle_filename)
    chunks = FixedSizeChunks(backend, max_variants_per_chunk=args.chunk_size)

    # Recover the validation indices.
    val_indices = np.load(VAL_INDICES_FILENAME)["arr_0"]
    train_indices = np.setdiff1d(np.arange(len(backend)), val_indices)

    full_dataset = chunks.get_dataset_for_chunk_id(args.chunk_index)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print("n train:", len(train_dataset))
    print("n val:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
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
        activation=config["model/activation"],
        use_standardized_genotype=config["use_standardized_genotype"],
        softmax_weights=torch.Tensor([1, 3, 1])
    )
    print(model)

    logger = None
    if args.wandb_logger:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(
            project=f"block_trainer_block_{args.chunk_index}"
        )
        logger = wandb_logger

    results_base = "chunk_checkpoints"
    if not os.path.isdir(results_base):
        os.makedirs(results_base)

    checkpoint_filename = f"model-block-{args.chunk_index}"

    model_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor=f"val_{OBJECTIVE}",
        mode="min",
        dirpath=results_base,
        filename=checkpoint_filename
    )

    callbacks = [
        # StopIfNan(),
        model_checkpoint,
        # EarlyStopping(
        #     monitor=f"val_{OBJECTIVE}",
        #     mode="min",
        #     patience=20
        # ),
    ]

    if ORION_SWEEP:
        callbacks.append(OrionCallback(f"val_{OBJECTIVE}"))

    trainer = pl.Trainer(
        log_every_n_steps=1,
        accelerator="gpu",
        devices=-1,  # use all GPUs
        max_epochs=config["max_epochs"],
        logger=logger,
        callbacks=callbacks
    )
    trainer.fit(model, train_loader, val_loader)


def main():
    args = parse_args()
    backend = BACKENDS[args.backend].load(args.backend_pickle_filename)
    chunks = FixedSizeChunks(backend, max_variants_per_chunk=args.chunk_size)

    if args.mode == "train":
        return train(args)

    # Create the validation dataset.
    n_val = int(round(len(backend) * args.val_proportion))
    val_indices = np.random.choice(len(backend), size=n_val, replace=False)
    if os.path.isfile(VAL_INDICES_FILENAME):
        raise IOError(f"File {VAL_INDICES_FILENAME} already exists.")

    np.savez_compressed(VAL_INDICES_FILENAME, val_indices)

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

    # Validation datset proportion.
    parser.add_argument(
        "--val-proportion",
        default=0.1,
        type=float,
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
