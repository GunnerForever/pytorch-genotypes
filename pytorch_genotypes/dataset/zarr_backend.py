"""
Zarr backend for pytorch genetic datasets.
"""


from typing import Set, Iterable, Optional, List, Tuple

from tqdm import tqdm
import torch
import zarr
import numpy as np
from numpy.typing import DTypeLike
from geneparse.core import GenotypesReader, Variant

from .core import GeneticDatasetBackend
from .utils import get_selected_samples_and_indexer, VariantPredicate


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
        filename_prefix: str,
        keep_samples: Optional[Set[str]] = None,
        variant_predicates: Optional[Iterable[VariantPredicate]] = None,
        chunks: Tuple[int, int] = (100_000, 10_000),
        dtype: DTypeLike = np.float16,
        progress: bool = True
    ):
        self.prefix = filename_prefix
        self.variants: List[Variant] = []

        self.samples, self._idx = get_selected_samples_and_indexer(
            reader, keep_samples
        )

        self.create_zarr(reader, variant_predicates, chunks, dtype, progress)
        self.cache = ZarrCache(self.z)

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
        reader: GenotypesReader,
        predicates: Optional[Iterable[VariantPredicate]],
        chunks: Tuple[int, int],
        dtype: DTypeLike,
        progress: bool
    ):
        """Create the zarr array from the genotypes reader."""
        # We initialize the zarr file with the dimensions of the geneparse
        # reader. It is likely that the final number of variants will be
        # smaller because of variants failing a predicate. In that case the
        # zarr array will be resized.
        if self.prefix.endswith(".zarr"):
            zarr_filename = self.prefix
        else:
            zarr_filename = f"{self.prefix}.zarr"

        self._zarr_filename = zarr_filename

        n_variants = reader.get_number_variants()
        n_samples = self.get_n_samples()
        z = zarr.open(
            zarr_filename,
            mode="w",
            shape=(n_samples, n_variants),
            chunks=chunks,
            dtype=dtype
        )
        variants = []

        # We target using ~8GB of memory max. Making this configurable would
        # be nice. Assuming geneparse uses float64.
        buf_size = round(8e9 / (n_samples * 8))

        with VariantBufferedZarrWriter(z, buf_size, dtype) as zarr_writer:
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

        # Resize to match the number of variants.
        z.resize(n_samples, len(variants))

        self.z = z
        self.variants = variants

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("z")
        d.pop("cache")
        return d

    def __setstate__(self, state):
        super().__setstate__(state)
        self.z = zarr.open(self._zarr_filename)
        self.cache = ZarrCache(self.z)

    def __getitem__(self, idx):
        return torch.tensor(self.cache.get_zarr_row(idx))

    def extract_range(self, left: int, right: int) -> torch.Tensor:
        return self.z.oindex[:, left:(right+1)]


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
        self.z[:, self.cur_j:right_boundary] = mat

        self.cur_j = right_boundary

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.buffer:
            self.flush_buffer()
