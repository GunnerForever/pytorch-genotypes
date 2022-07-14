from typing import Optional

import torch
from torch import Tensor
from torch.distributions.geometric import Geometric
import torch.nn as nn


class ChunkDropout(nn.Module):
    def __init__(
        self,
        input_length: int,
        dropout_probability: float,
        hole_length_loc: int,
        hole_length_scale: int,
        minimum_hole_length: Optional[int] = None
    ):
        """Module that applies dropout in chunks.

        This is meant to be used for input dropout of genetic variant data so
        that adjacent variants are dropped out together.

        Arguments:

            input_length: Number of features (e.g. variants).
            dropout_probability: Probability of starting a dropout chunk at a
                                 given variant.
            hole_length_loc: Mean hole lenth (sampled from a discretized
                             bounded normal variable.)
            hole_length_scale: Standard deviation of the hole length
                               distribution.
            minimum_hole_length: Set to 1 by default.

        """
        super().__init__()
        self.input_length = input_length
        self.dropout_probability = dropout_probability
        self.hole_length_loc = hole_length_loc
        self.hole_length_scale = hole_length_scale

        if minimum_hole_length is None:
            self.minimum_hole_length = torch.tensor(1)

        else:
            self.minimum_hole_length = torch.tensor(minimum_hole_length)

    def get_dropout_indices(self, _verbose: bool = False) -> Tensor:
        output = torch.zeros(self.input_length, dtype=torch.bool)

        last_gap = (0, 0)

        while True:
            # Sample a start position (counting from the end of the last gap).
            new_gap_offset = Geometric(self.dropout_probability).sample()
            if new_gap_offset == 0:
                new_gap_offset += 1

            gap_start = last_gap[1] + new_gap_offset

            if gap_start >= self.input_length - 1:
                if _verbose:
                    print()

                return output

            # Sample gap length.
            gap_length = torch.normal(
                mean=float(self.hole_length_loc),
                std=float(self.hole_length_scale),
                size=(1, )
            ).to(torch.int)

            if gap_length < self.minimum_hole_length:
                gap_length = self.minimum_hole_length

            gap_end = gap_start + gap_length
            if gap_end > self.input_length:
                gap_end = self.input_length

            last_gap = (int(gap_start), int(gap_end))

            if _verbose:
                print(last_gap, end=", ")

            output[last_gap[0]:last_gap[1]] = True

            if gap_end >= self.input_length:
                if _verbose:
                    print()
                return output

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            x[:, self.get_dropout_indices()] = 0
            return x

        return x


def visualize_chunk_dropout(n, *args, **kwargs):
    """Simulates n different applications of the chunk dropout module.

    Example to show dropout on a matrix with 1000 SNPs represented as noise for
    visualization purposes:

    visualize_chunk_dropout(100, 1000, 0.01, 50, 50)

    The first argument is the number of simulated batches and only impacts
    what the visualization will look like.

    The above examples takes the sequence length to be 1000, the probability of
    opening a dropout chunk 1% and the mean and standard deviation of the
    dropout chunk length to be 50.

    """
    import matplotlib.pyplot as plt

    mod = ChunkDropout(*args, **kwargs)

    matrices = []
    for _ in range(n):
        cur = torch.rand(20, mod.input_length)
        cur = mod(cur)
        matrices.append(cur)

    plt.matshow(torch.vstack(matrices).numpy())
    plt.colorbar()
    plt.show()
