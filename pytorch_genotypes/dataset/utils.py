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


def standardize_features(
    matrix: TorchOrNumpyArray,
    impute_to_mean=False
) -> Tuple[TorchOrNumpyArray, TorchOrNumpyArray, TorchOrNumpyArray]:
    """Standardize a design matrix of n_samples x n_features.

    The return type is the standardized matrix, the vector used for centering
    and the vector used for scaling (the standard deviation).

    TODO make better type annotations, e.g. using @overload.

    """
    center = np.nanmean(matrix, axis=0)
    scale = np.nanstd(matrix, axis=0)

    matrix -= center
    matrix /= scale

    if impute_to_mean:
        matrix = np.nan_to_num(matrix, nan=0)

    return (matrix, center, scale)


def rescale_standardized(
    matrix: TorchOrNumpyArray,
    center: TorchOrNumpyArray,
    scale: TorchOrNumpyArray,
) -> TorchOrNumpyArray:
    """Inverse operation of standardize_features(...)."""
    return matrix * scale + center


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
