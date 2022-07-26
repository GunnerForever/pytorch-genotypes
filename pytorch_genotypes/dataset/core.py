"""
Abstract classes for datasets and dataset backends.

The idea is that the backends provide access to genetic data wheareas datasets
can implement additional logic, for example to include phenotype data.

"""

import pickle
from collections import defaultdict
from typing import List, Type, TypeVar, Tuple, TYPE_CHECKING, Optional

import torch
from geneparse import Variant

from torch.utils.data.dataset import Dataset


if TYPE_CHECKING:
    import pandas as pd


T = TypeVar("T", bound="GeneticDatasetBackend")


class GeneticDatasetBackend(object):
    def get_samples(self) -> List[str]:
        raise NotImplementedError()

    def get_variants(self) -> List[Variant]:
        raise NotImplementedError()

    def get_n_samples(self) -> int:
        raise NotImplementedError()

    def get_n_variants(self) -> int:
        raise NotImplementedError()

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

    @classmethod
    def load(cls: Type[T], filename: str) -> T:
        with open(filename, "rb") as f:
            o = pickle.load(f)

        assert isinstance(o, cls)
        return o

    def __getitem__(self, idx) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.get_n_samples()


class GeneticDataset(Dataset):
    def __init__(self, backend: GeneticDatasetBackend):
        super().__init__()
        self.backend = backend

    def load_full_dataset(self) -> Tuple[torch.Tensor, ...]:
        """Utility function to load everything in memory.

        This is useful for testing datasets or to use with models that don't
        train by minibatch.

        """
        tensors = defaultdict(list)

        for i in range(len(self)):
            datum: Tuple[torch.Tensor, ...] = self[i]

            for j in range(len(datum)):
                tensors[j].append(datum[j])

        # Merge everything.
        return tuple((
            torch.vstack(tensors[j]) for j in range(len(tensors))
        ))

    def __getitem__(self, idx):
        return self.backend[idx]

    def __len__(self) -> int:
        return len(self.backend)


class _Chunk(object):
    __slots__ = ("id", "first_variant_index", "last_variant_index")

    def __init__(
        self,
        id: int,
        first_variant_index: Optional[int],
        last_variant_index: Optional[int]
    ):
        self.id = id
        self.first_variant_index = first_variant_index
        self.last_variant_index = last_variant_index

    def __repr__(self):
        return (
            f"<Chunk #{self.id} - "
            f"{self.first_variant_index}:{self.last_variant_index}"
        )


class FixedSizeChunks(object):
    """Splits the variants in a backend into contiguous chunks.

    This class mostly abstracts away index arithmetic.

    """
    def __init__(
        self,
        backend: GeneticDatasetBackend,
        max_variants_per_chunk=2000,
    ):
        self.backend = backend
        self.chunks: List[_Chunk] = []
        self.max_variants_per_chunk = max_variants_per_chunk

        self._load_chunks()

    def _load_chunks(self):
        """Assigns the variants in the backend to chunks."""
        variants = self.backend.get_variants()
        n = len(variants)

        max_variants_per_chunk = self.max_variants_per_chunk

        def _chunk_generator():
            left = 0
            cur_id = 0

            while left < n:
                # Check if the right boundary is on the same chromosome.
                # The step size is either to the last variant or by
                # max_variants_per_chunk.
                cur_chrom = variants[left].chrom
                right = min(left + max_variants_per_chunk - 1, n - 1)

                if cur_chrom != variants[right].chrom:
                    # We need to go back and search for the largest right bound
                    # on the same chromosome.
                    right = left + 1
                    while variants[right].chrom == cur_chrom:
                        right += 1

                    right -= 1

                yield _Chunk(cur_id, left, right)
                cur_id += 1
                left = right + 1

        self.chunks = list(_chunk_generator())

    def get_chunk_id_for_variant(self, v: Variant) -> int:
        # This is TODO
        raise NotImplementedError()
        return -1

    def get_chunk(self, chunk_id: int) -> _Chunk:
        return self.chunks[chunk_id]

    def __len__(self) -> int:
        return len(self.chunks)

    def get_variants_for_chunk_id(self, chunk_id: int) -> List[Variant]:
        variants = self.backend.get_variants()

        chunk = self.get_chunk(chunk_id)

        assert chunk.first_variant_index is not None
        assert chunk.last_variant_index is not None

        return variants[chunk.first_variant_index:(chunk.last_variant_index+1)]

    def get_variant_dataframe_for_chunk_id(
        self,
        chunk_id: int
    ) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame(
            [
                (o.name, o.chrom.name, o.alleles[0], ",".join(o.alleles[1:]))
                for o in self.get_variants_for_chunk_id(chunk_id)
            ],
            columns=["name", "chrom", "pos", "allele1", "allele2"]
        )

    def get_tensor_for_chunk_id(self, chunk_id: int) -> torch.Tensor:
        chunk = self.get_chunk(chunk_id)
        assert chunk.last_variant_index is not None
        return self.backend[
            :,
            chunk.first_variant_index:(chunk.last_variant_index+1)
        ]
