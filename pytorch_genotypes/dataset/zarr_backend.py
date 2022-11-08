"""
Zarr backend for pytorch genetic datasets.
"""


import os
import pickle
from typing import Iterable, List, Optional, Set, Tuple, Sized, Iterator

import numpy as np
import torch
from torch.utils.data import Sampler
import zarr
from zarr.convenience import PathNotFoundError
from geneparse.core import GenotypesReader, Variant
from numpy.typing import DTypeLike
from tqdm import tqdm

from .core import GeneticDatasetBackend, GeneticDataset, MaskBackendWrapper
from .utils import (VariantPredicate, get_selected_samples_and_indexer,
                    resolve_path)


class ZarrCache:
    """Memory cache to make reads from sequential indexes faster."""
    def __init__(self, z: zarr.Array):
        self._chunk_rows = z.chunks[0]
        self._chunk_cur = None
        self._chunk_data: Optional[torch.Tensor] = None
        self.z = z

    def get_zarr_row(self, i: int) -> torch.Tensor:
        """Cache chunk and return requested row."""
        self._cache_chunk(i)
        assert self._chunk_data is not None
        return self._chunk_data[i % self._chunk_rows, :]

    def _cache_chunk(self, i: int) -> None:
        chunk_id = i // self._chunk_rows
        if self._chunk_cur == chunk_id:
            # The chunk is currently cached, do nothing.
            return

        # Fetch and cache the chunk.
        left = chunk_id * self._chunk_rows
        right = left + self._chunk_rows
        self._chunk_cur = chunk_id
        self._chunk_data = torch.tensor(
            self.z.oindex[left:right, slice(None)]
        )


class ZarrBackend(GeneticDatasetBackend):
    def __init__(
        self,
        reader: GenotypesReader,
        prefix: str,
        zarr_root: Optional[zarr.Group] = None,
        keep_samples: Optional[Set[str]] = None,
        variant_predicates: Optional[Iterable[VariantPredicate]] = None,
        chunks: Tuple[int, int] = (100_000, 10_000),
        dtype: DTypeLike = np.float16,
        progress: bool = True
    ):
        self.prefix = prefix
        self.variants: List[Variant] = []

        self.samples, self._idx = get_selected_samples_and_indexer(
            reader, keep_samples
        )

        self.create_zarr(
            zarr_root, reader, variant_predicates, chunks, dtype, progress
        )
        self.cache = ZarrCache(self._z)

    def get_samples(self):
        return self.samples

    def get_variants(self):
        return self.variants

    def get_n_samples(self):
        return len(self.samples)

    def get_n_variants(self):
        return len(self.variants)

    def create_zarr(
        self,
        zarr_root,
        reader: GenotypesReader,
        predicates: Optional[Iterable[VariantPredicate]],
        chunks: Tuple[int, int],
        dtype: DTypeLike,
        progress: bool
    ):
        """Create the zarr array from the genotypes reader."""
        if zarr_root is None:
            self._create_zarr_local(reader, chunks, dtype)
        else:
            self._create_zarr_in_group(reader, zarr_root, chunks, dtype)

        self._fill_array(reader, predicates, dtype, progress)

    def _create_zarr_in_group(self, reader, zarr_root, chunks, dtype):
        self._local = False
        n_samples = self.get_n_samples()
        n_variants = reader.get_number_variants()
        self._z = zarr_root.create_dataset(
            self.prefix,
            shape=(n_samples, n_variants),
            chunks=chunks,
            dtype=dtype
        )

    def _create_zarr_local(self, reader, chunks, dtype):
        self._local = True
        n_samples = self.get_n_samples()
        n_variants = reader.get_number_variants()

        if self.prefix.endswith(".zarr"):
            zarr_filename = self.prefix
        else:
            zarr_filename = f"{self.prefix}.zarr"

        self._zarr_filename = zarr_filename
        self._z = zarr.open(
            zarr_filename,
            mode="w",
            shape=(n_samples, n_variants),
            chunks=chunks,
            dtype=dtype
        )

    def _fill_array(self, reader, predicates, dtype, progress):
        variants = []
        # We target using ~6GB of memory max. Making this configurable would
        # be nice. Assuming geneparse uses float64.
        n_samples = self.get_n_samples()
        buf_size = round(6e9 / (n_samples * 8))

        with VariantBufferedZarrWriter(self._z, buf_size, dtype) as zarr_writer:  # noqa: E501
            if progress:
                iterator = tqdm(
                    reader.iter_genotypes(),
                    total=reader.get_number_variants()
                )
            else:
                iterator = reader.iter_genotypes()

            for g in iterator:
                # Check if filtered out by any of the predicates.
                if any([not f(g) for f in predicates or []]):
                    continue

                variants.append(g.variant)
                if self._idx is None:
                    zarr_writer.add(g.genotypes)
                else:
                    zarr_writer.add(g.genotypes[self._idx])

        # We initialize the zarr file with the dimensions of the geneparse
        # reader. It is likely that the final number of variants will be
        # smaller because of variants failing a predicate. In that case the
        # zarr array will be resized.
        self._z.resize(n_samples, len(variants))
        self.variants = variants

    @classmethod
    def load(cls, filename: str) -> "ZarrBackend":
        with open(filename, "rb") as f:
            backend = pickle.load(f)

        if backend._local:
            contexts = [
                os.curdir,
                os.path.dirname(filename)
            ]

            zarr_fn = resolve_path(backend._zarr_filename, contexts=contexts)
            backend._zarr_filename = zarr_fn
            backend._z = zarr.open(backend._zarr_filename)
            backend.cache = ZarrCache(backend._z)

        return backend

    def __getstate__(self):
        d = self.__dict__.copy()
        if self._local:
            d.pop("_z")

        d.pop("cache")
        return d

    def __setstate__(self, state):
        super().__setstate__(state)
        try:
            if self._local:
                self._z = zarr.open(self._zarr_filename, mode="r")

            self.cache = ZarrCache(self._z)
        except PathNotFoundError:
            self._z = None
            self.cache = None

    def __getitem__(self, idx) -> torch.Tensor:
        return self.cache.get_zarr_row(idx)

    def extract_range(self, left: int, right: int) -> torch.Tensor:
        return self._z.oindex[:, left:(right+1)]


class VariantBufferedZarrWriter(object):
    def __init__(self, z, buf_size, dtype):
        self.cur_j = 0
        self.buf_size = buf_size
        self.buffer = []
        self.z = z
        self.dtype = dtype

    def add(self, genotypes: np.ndarray):
        self.buffer.append(genotypes)

        if len(self.buffer) >= self.buf_size:
            self.flush_buffer()

    def flush_buffer(self):
        mat = np.vstack(self.buffer).astype(self.dtype).T
        self.buffer = []
        right_boundary = self.cur_j + mat.shape[1]
        self.z.oindex[:, self.cur_j:right_boundary] = mat

        self.cur_j = right_boundary

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.buffer:
            self.flush_buffer()


class ZarrCacheAwareSampler(Sampler):
    """Sampler subclass that will sample within a chunk at a time.
    This takes advantage of the Zarr row-wise caching.
    """
    def __init__(self, dataset: GeneticDataset):
        self.dataset = dataset

        backend: GeneticDatasetBackend = self.dataset.backend

        if isinstance(backend, MaskBackendWrapper):
            # Check if samples have been subset.
            if backend.samples_keep_indices is not None:
                raise ValueError(
                    "Can't use CacheAwareSampler if samples have been masked. "
                    "To achieve this with reasonable performance, you need to "
                    "filter samples at backend creation."
                )
            else:
                backend = backend.backend

        if not isinstance(backend, ZarrBackend):
            raise ValueError(type(self.dataset.backend))

        self.chunk_size = backend._z.chunks[0]

    def __iter__(self) -> Iterator:
        cur_chunk = 0
        n = len(self)
        stop = False
        while not stop:
            left = cur_chunk * self.chunk_size

            if left + self.chunk_size > n:
                max_offset = n - left
                stop = True
            else:
                max_offset = self.chunk_size

            permutation = torch.randperm(max_offset)
            for offset in permutation:
                yield left + offset

    def __len__(self) -> int:
        return len(self.dataset)
