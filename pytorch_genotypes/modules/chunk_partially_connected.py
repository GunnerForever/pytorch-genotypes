from typing import Optional, Callable

import torch
import torch.nn as nn


class ChunkPartiallyConnected(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,  # Per-chunk output size.
        chunk_size: int = 1000,
        module_f: Optional[Callable[[int], nn.Module]] = None,
        chunk_the_input: bool = True
    ):
        super().__init__()

        self.out_features = out_features
        self.chunk_size = chunk_size
        self.chunk_the_input = chunk_the_input

        if module_f is None:
            if chunk_the_input:
                module_f = lambda n_in: nn.Linear(  # noqa: E731
                    n_in, out_features
                )
            else:
                module_f = lambda n_out: nn.Linear(  # noqa: E731
                    in_features, n_out
                )

        sequence_length = in_features if chunk_the_input else out_features

        self.bounds = []
        modules = []
        for left, right in _create_chunks(sequence_length, chunk_size):
            # Create the subnetwork.
            subnet = module_f(right - left)

            self.bounds.append((left, right))
            modules.append(subnet)

        self.submodules = nn.ModuleList(modules)

    def get_effective_output_size(self):
        return sum((o.out_features for o in self.submodules))

    def forward(self, input):
        if self.chunk_the_input:
            return self._forward_chunked_input(input)
        else:
            return self._forward_chunked_output(input)

    def _forward_chunked_input(self, input):
        chunks = []
        for (left, right), module in zip(self.bounds, self.submodules):
            chunks.append(module(input[:, left:right]))

        return torch.hstack(tuple(chunks))

    def _forward_chunked_output(self, input):
        outputs = []
        for module in self.submodules:
            outputs.append(module(input))

        return torch.hstack(tuple(outputs))


def _create_chunks(n_features, chunk_size):
    i = chunk_size
    # If the right bound falls outside the sequence stop.
    while i <= n_features:
        yield ((i-chunk_size), i)
        i += chunk_size

    # Check if the left bound is in the sequence.
    if i-chunk_size < n_features:
        yield (i-chunk_size, n_features)
