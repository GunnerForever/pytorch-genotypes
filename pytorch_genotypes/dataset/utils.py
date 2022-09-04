from typing import Optional, Set, Callable, Tuple, Union, List
import os

import torch
import numpy as np
from geneparse.core import GenotypesReader, Genotypes


VariantPredicate = Callable[[Genotypes], bool]
TorchOrNumpyArray = Union[np.ndarray, torch.Tensor]


def resolve_path(filename: str, contexts: Optional[List[str]] = None) -> str:
    """Tries to resolve a path, possibly in different locations (contexts)."""
    try:
        with open(filename, "r"):
            pass
        return filename
    except FileNotFoundError:
        pass

    if contexts:
        for context in contexts:
            cur = os.path.join(context, filename)
            try:
                with open(cur, "r"):
                    pass
                return cur
            except FileNotFoundError:
                pass

    raise FileNotFoundError(filename)


class TensorScaler(object):
    """Standardize a tensor of n_samples x n_features."""
    def __init__(self, tensor: torch.Tensor):
        tensor = tensor.to(torch.float32)

        self.center = torch.nanmean(tensor, dim=0)
        self.scale = torch.sqrt(
            torch.nanmean((tensor - self.center) ** 2, dim=0)
        )

    def standardize_tensor(self, t: torch.Tensor):
        return (t - self.center) / self.scale


def get_selected_samples_and_indexer(
    reader: GenotypesReader,
    keep_samples: Optional[Set[str]]
):
    """Utility function to overlap geneparse samples with selected IDs.

    This function returns the list of samples and their corresponding indices
    in the genotype file as a vector of ints suitable for indexing.

    """
    file_samples = reader.get_samples()

    if keep_samples is None:
        return file_samples, None

    file_samples_set = set(file_samples)
    overlap = file_samples_set & keep_samples

    genetic_sample_type = type(file_samples[0])
    keep_samples_type = type(next(iter(keep_samples)))

    if genetic_sample_type is not keep_samples_type:
        raise ValueError(
            f"Genetic file sample type: '{genetic_sample_type}' is "
            f"different from provided samples list ('{keep_samples_type}'"
            ")."
        )

    if len(overlap) == 0:
        raise ValueError(
            "No overlap between keep_samples and genetic dataset."
        )

    indices = []
    samples = []
    for index, sample in enumerate(file_samples):
        if sample in keep_samples:
            samples.append(sample)
            indices.append(index)

    return samples, np.array(indices, dtype=int)
