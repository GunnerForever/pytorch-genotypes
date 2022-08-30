#!/usr/bin/env python


import argparse
from math import ceil

from tqdm import tqdm
import torch
from pytorch_genotypes.models import GenotypeProbabilisticAutoencoder
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)

    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--batch-size", default=256, type=int)

    return parser.parse_args()


def main(args):
    device = torch.device(args.device)

    ae = GenotypeProbabilisticAutoencoder.load_from_checkpoint(args.checkpoint)
    ae = ae.to(device)

    n_snps = ae.decoder[-1].out_features

    embeddings = []

    chunk_size = args.batch_size
    input_generator = identity_matrix_chunks(n_snps, chunk_size, device=device)

    for x in tqdm(input_generator, total=ceil(n_snps / chunk_size)):
        cur_embeddings = ae.encoder(x).cpu().detach().numpy()
        embeddings.append(cur_embeddings)

    embeddings = np.vstack(embeddings)
    np.savez_compressed("_embeddings.npz", embeddings)


def identity_matrix_chunks(n_columns, batch_size, device=None):
    """Yields row chunks of an identity matrix with n_columns."""
    cur = 0
    identity = torch.eye(batch_size, dtype=torch.float32, device=device)

    while True:
        # Left part.
        to_stack = []
        left_padding = cur
        if left_padding > 0:
            left = torch.zeros(batch_size, left_padding, device=device)
            to_stack.append(left)

        right_padding = n_columns - (cur + batch_size)
        if right_padding > 0:
            right = torch.zeros(batch_size, right_padding, device=device)
            to_stack.append(identity)
            to_stack.append(right)
        elif right_padding < 0:
            to_stack.append(identity[:, :right_padding])
        else:
            to_stack.append(identity)

        stacked = torch.hstack(tuple(to_stack))

        if right_padding < 0:
            # We also need to reduce the number of rows.
            stacked = stacked[:right_padding, :]

        yield stacked
        cur += batch_size

        if cur >= n_columns:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
