import functools
import math
import os
from collections import OrderedDict
from typing import (Callable, Dict, Iterable, List, Optional, Set, TypeVar,
                    Union)

import numpy as np
import torch
from geneparse.core import Genotypes, GenotypesReader
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler

VariantPredicate = Callable[[Genotypes], bool]
TorchOrNumpyArray = Union[np.ndarray, torch.Tensor]


def resolve_path(filename: str, contexts: Optional[List[str]] = None) -> str:
    """Tries to resolve a path, possibly in different locations (contexts)."""
    try:
        with open(filename, "r"):
            pass
        return filename
    except IsADirectoryError:
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
            except IsADirectoryError:
                return cur
            except FileNotFoundError:
                pass

    raise FileNotFoundError(filename)


def chunk_op_by_col(
    matrices: Union[torch.Tensor, Iterable[torch.Tensor]],
    f_map: Callable[[List[torch.Tensor]], torch.Tensor],
    chunk_size=10_000
):

    if isinstance(matrices, torch.Tensor):
        matrices = [matrices]
    else:
        matrices = list(matrices)

    n_col = matrices[0].shape[1]
    assert chunk_size < n_col

    # Ensure that if many matrices are passed, they all have the same number of
    # columns.
    assert len(set([mat.shape[1] for mat in matrices])) == 1

    results = torch.empty(n_col, dtype=matrices[0].dtype)

    i = 0
    r = chunk_size
    while r - chunk_size < n_col:
        cur_matrices = [
            mat[:, (r-chunk_size):min(r, n_col)] for mat in matrices
        ]

        r += chunk_size
        cur_results = f_map(cur_matrices)

        cur_chunk_size = len(cur_results)
        results[i:(i+cur_chunk_size)] = cur_results
        i += cur_chunk_size

    return results


def tensor_chunk_mean(
    mat: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    return chunk_op_by_col(
        mat,
        lambda chunk: torch.nanmean(chunk[0], dim=0),
        chunk_size=chunk_size
    )


class TensorScaler(object):
    """Standardize a tensor of n_samples x n_features."""
    def __init__(
        self,
        tensor: torch.Tensor,
        max_memory_in_bytes: int = int(1e8),
        from_mafs: Optional[torch.Tensor] = None,
        _force_chunk_size: int = 0
    ) -> None:
        tensor = tensor.to(torch.float32)

        if from_mafs is None:
            self.init_scaler_default(tensor, max_memory_in_bytes,
                                     _force_chunk_size)

        else:
            self.init_scaler_from_mafs(from_mafs)

    def init_scaler_from_mafs(self, mafs: torch.Tensor):
        self.center = 2 * mafs
        self.scale = torch.sqrt(2 * mafs * (1 - mafs))

    def init_scaler_default(
        self,
        tensor: torch.Tensor,
        max_memory_in_bytes: int,
        _force_chunk_size: int
    ):
        tensor_bytes = tensor.element_size() * tensor.nelement()

        if _force_chunk_size or (tensor_bytes > max_memory_in_bytes):
            # Find an appropriate chunk size.
            if _force_chunk_size != 0:
                chunk_size = _force_chunk_size
            else:
                chunk_size = math.floor(
                    max_memory_in_bytes /
                    (tensor.element_size() * tensor.shape[0])
                )
                chunk_size = max(1, chunk_size)
                print("Will use chunk_size of ", chunk_size)

            mean: Callable[[torch.Tensor], torch.Tensor] = functools.partial(
                tensor_chunk_mean,
                chunk_size=chunk_size
            )

            def std(tensor, center):
                return chunk_op_by_col(
                    [tensor, center.reshape(1, -1)],
                    f_map=lambda matrices: torch.sqrt(torch.nanmean(
                        (matrices[0] - matrices[1]) ** 2, dim=0
                    )),
                    chunk_size=chunk_size
                )

        else:
            mean = functools.partial(torch.nanmean, dim=0)

            def std(tensor, center):
                return torch.sqrt(mean((tensor - center) ** 2))

        self.center = mean(tensor)
        self.scale = std(tensor, self.center)

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


EnvLabel = TypeVar("EnvLabel")


class MultiEnvironmentIteratorFromMultipleDatasets(object):
    def __init__(
        self,
        datasets: List[Dataset],
        batch_size: int = 1,
        sampler_from_dataset: Optional[Callable[[Dataset], Sampler]] = None,
        **kwargs
    ):
        dataloaders = []
        for dataset in datasets:
            # Allow custom samplers created from a callback.
            if sampler_from_dataset is not None:
                sampler: Optional[Sampler] = sampler_from_dataset(dataset)
            elif "sampler" in kwargs:
                sampler = kwargs["sampler"]
            else:
                sampler = None

            dl = DataLoader(dataset, batch_size, sampler=sampler, **kwargs)
            dataloaders.append(dl)

        self.dataloaders = dataloaders
        self.batch_size = batch_size

        self.iterators = []
        for dl in self.dataloaders:
            self.iterators.append(iter(dl))

    def __iter__(self):
        out = []

        for i in range(len(self.iterators)):
            try:
                batch_li = next(self.iterators[i])
            except StopIteration:
                self.iterators[i] = iter(self.dataloaders[i])
                batch_li = next(self.iterators[i])

            out.append(batch_li)

        yield out


class MultiEnvironmentIteratorFromSingleDataset(object):
    def __init__(
        self,
        dataset: Dataset,
        env_to_indices: Dict[EnvLabel, List[int]],
        batch_size: int = 1,
        **kwargs
    ):
        # Create a dataloader per env by creating subsets.
        dataloaders = OrderedDict()
        for env in env_to_indices.keys():
            indices = env_to_indices[env]
            if max(indices) >= len(dataset):  # type: ignore
                raise IndexError

            sampler = SubsetRandomSampler(indices)
            dl = DataLoader(dataset, batch_size, shuffle=False,
                            sampler=sampler, **kwargs)
            dataloaders[env] = dl

        self.dataloaders = dataloaders
        self.env_to_indices = env_to_indices
        self.batch_size = batch_size

        # Prime the iterators.
        self.iterators = {}
        for env, dl in self.dataloaders.items():
            self.iterators[env] = iter(dl)

    def __iter__(self):
        out = []

        for env in self.iterators.keys():
            try:
                batch_li = next(self.iterators[env])
            except StopIteration:
                self.iterators[env] = iter(self.dataloaders[env])
                batch_li = next(self.iterators[env])

            out.append(batch_li)

        yield out
