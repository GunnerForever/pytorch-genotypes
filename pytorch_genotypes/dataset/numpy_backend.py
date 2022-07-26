"""
Numpy backend for genotypes for datasets that fit in memory.
"""

import os
import pickle
from typing import Optional, Set, Iterable, List

from tqdm import tqdm
import torch
import numpy as np
from numpy.typing import DTypeLike
from geneparse.core import GenotypesReader, Variant

from .core import GeneticDatasetBackend
from .utils import (
    resolve_path,
    get_selected_samples_and_indexer,
    VariantPredicate,
)


class NumpyBackend(GeneticDatasetBackend):
    def __init__(
        self,
        reader: GenotypesReader,
        npz_filename: str,
        keep_samples: Optional[Set[str]] = None,
        variant_predicates: Optional[Iterable[VariantPredicate]] = None,
        dtype: DTypeLike = np.float16,
        impute_to_mean: bool = True,
        progress: bool = True,
        use_absolute_path: bool = False,
        lazy_pickle: bool = False,
    ):
        self.samples, self._idx = get_selected_samples_and_indexer(
            reader, keep_samples
        )

        self.lazy_pickle = lazy_pickle
        self.use_absolute_path = use_absolute_path

        if use_absolute_path:
            self.npz_filename = os.path.abspath(npz_filename)
        else:
            self.npz_filename = npz_filename

        self.variants: List[Variant] = []

        self._create_np_matrix(reader, variant_predicates, dtype,
                               impute_to_mean, progress)

    def _create_np_matrix(
        self,
        reader: GenotypesReader,
        variant_predicates: Optional[Iterable[VariantPredicate]],
        dtype: DTypeLike,
        impute_to_mean: bool,
        progress: bool
    ):
        n_variants = reader.get_number_variants()

        m = np.empty((self.get_n_samples(), n_variants), dtype=dtype)
        cur_column = 0
        variants = []
        if progress:
            iterator = tqdm(reader.iter_genotypes(), total=n_variants)
        else:
            iterator = reader.iter_genotypes()

        for g in iterator:
            if any([not f(g) for f in variant_predicates or []]):
                continue

            if self._idx is not None:
                genotypes = g.genotypes[self._idx]
            else:
                genotypes = g.genotypes

            if impute_to_mean:
                mean = np.nanmean(genotypes)
                genotypes[np.isnan(genotypes)] = mean

            m[:, cur_column] = genotypes.astype(dtype)
            variants.append(g.variant)
            cur_column += 1

        # Resize if some variants were filtered out.
        m = m[:, :cur_column]

        self.variants = variants
        self.m: Optional[np.ndarray] = m
        np.savez_compressed(self.npz_filename, m)

    @classmethod
    def load(cls, filename: str) -> "NumpyBackend":
        with open(filename, "rb") as f:
            o = pickle.load(f)

        lazy_pickle = getattr(o, "lazy_pickle", False)
        use_absolute_path = getattr(o, "use_absolute_path", False)

        if not lazy_pickle:
            if use_absolute_path:
                contexts = []
            else:
                contexts = [
                    os.curdir,
                    os.path.dirname(filename)
                ]

            filename = resolve_path(o.npz_filename, contexts=contexts)
            o.m = np.load(filename)["arr_0"]

        return o

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("m")
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        self.m = None

    def __getitem__(self, idx):
        if self.m is None:
            # Try to lazily load the numpy matrix.
            self.m = np.load(resolve_path(self.npz_filename))["arr_0"]

        return torch.tensor(self.m[idx, :])

    def get_samples(self):
        return self.samples

    def get_variants(self):
        return self.variants

    def get_n_samples(self):
        return len(self.samples)

    def get_n_variants(self):
        return len(self.variants)
