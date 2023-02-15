"""
Abstract classes for datasets and dataset backends.

The idea is that the backends provide access to genetic data wheareas datasets
can implement additional logic, for example to include phenotype data.

"""

import logging
import pickle
import itertools
import collections
from collections import defaultdict, OrderedDict
from typing import (
    List, Type, TypeVar, Tuple, TYPE_CHECKING, Optional, Union, Any,
    Iterable
)

import torch
import numpy as np
from geneparse import Variant
from torch.utils.data.dataset import Dataset

from .utils import TensorScaler


if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)


T = TypeVar("T", bound="GeneticDatasetBackend")


# Return a callable that takes *args -> named tuple like.
class Batch(collections.abc.Sequence):
    __slots__ = ("name", "_fields", "payload", "_key_to_index")

    def __init__(self, name: str, fields: Iterable[str], *payload) -> None:
        self.name = name
        self._fields = tuple(fields)
        self.payload: Tuple[Any, ...] = payload

        self._key_to_index = {k: i for i, k in enumerate(fields)}

    def __repr__(self) -> str:
        s = f"<{self.name} - "
        parts = []
        for field, value in zip(self._fields, self.payload):
            if isinstance(value, torch.Tensor):
                shape = "x".join([str(i) for i in value.shape])
                value = f"Tensor<{shape}>"

            parts.append(f"{field}={value}")

        return s + ", ".join(parts) + ">"

    def __iter__(self):
        for value in self.payload:
            yield value

    def __getattr__(self, __name: str) -> Any:
        return self.payload[self._key_to_index[__name]]

    def __getitem__(self, k: int) -> Any:  # type: ignore
        return self.payload[k]

    def __len__(self) -> int:
        return len(self._fields)


class BatchFactory(object):
    def __init__(self, name: str, fields: Tuple[str, ...]):
        self.name = name
        self.fields = fields

    def __call__(self, *payload) -> Batch:
        return Batch(self.name, self.fields, *payload)

    @classmethod
    def union(cls, name, *batches: "BatchFactory") -> "BatchFactory":
        fields = tuple(OrderedDict.fromkeys(
            itertools.chain(*[batch.fields for batch in batches])
        ))

        return BatchFactory(name, fields)


def batch_collate_by_standardized(batches: List[Batch]):
    if not hasattr(batches[0], "std_genotype"):
        raise ValueError("Current batch does not provide standardized "
                         "genotypes.")

    return torch.vstack([b.std_genotype for b in batches])


BatchDosage = BatchFactory("BatchDosage", ("dosage", ))


BatchDosageStandardized = BatchFactory(
    "BatchDosageStandardized", ("dosage", "std_genotype")
)


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

    def extract_range(
        self,
        left: int,
        right: int,
    ) -> torch.Tensor:
        """Extract variants in a range of indices.

        The default implementation is likely to be very inefficient because
        it loops over the rows. Subclasses should provide more efficient
        implementations.

        """
        logger.warning(
            "The extract_range method may be slow for generic backends."
        )

        out = torch.empty((len(self), right - left + 1))
        for i in range(len(self)):
            out[i, :] = self[i][left:(right+1)]

        return out


class GeneticDatasetBackendWrapper(GeneticDatasetBackend):
    """Class that mimics a backend but applies other forms of filtering or
    processing.

    For examples, backends that represent subsets of variants or individuals
    from a "parent" backend.

    Serialization would require recursion into the parent backend and would
    be fairly convoluted. For this reason, we don't support pickling of
    wrappers. However, they should be easy to recreate using the same
    parent backend and arguments.

    """
    def __getstate__(self) -> dict:
        raise NotImplementedError()

    def __setstate__(self, state: dict):
        raise NotImplementedError()


class GeneticDataset(Dataset):
    def __init__(
        self,
        backend: GeneticDatasetBackend,
        genotype_standardization: bool = True
    ):
        super().__init__()
        self.backend = backend
        if genotype_standardization:
            self.scaler: Union[TensorScaler, None] = self.create_scaler()
        else:
            self.scaler = None

    def create_scaler(self, max_n: int = 2000) -> TensorScaler:
        """Column-wise (genotype) scaler.

        This method is defined at the datset level because it is important
        not to use test data to estimate scaling to avoid information leakage.

        """
        # Estimate scaling on max 2k rows.
        n = len(self.backend)

        if n > max_n:
            indices = np.sort(np.random.choice(
                np.arange(n), size=max_n, replace=False
            ))
            n = max_n
        else:
            indices = np.arange(n)

        mat = torch.empty(
            (n, self.backend.get_n_variants()),
            dtype=torch.float32
        )

        for mat_index, index in enumerate(indices):
            mat[mat_index, :] = self.backend[index]

        return TensorScaler(mat)

    def load_full_dataset(self) -> Tuple[torch.Tensor, ...]:
        """Utility function to load everything in memory.

        This is useful for testing datasets or to use with models that don't
        train by minibatch.

        """
        tensors = defaultdict(list)

        for i in range(len(self)):
            datum: Batch = self[i]

            for j in range(len(datum)):
                tensors[j].append(datum[j])

        # Merge everything.
        return tuple((
            torch.vstack(tensors[j]) for j in range(len(tensors))
        ))

    def __getitem__(self, idx) -> Batch:
        """Genetic datasets provide the dosage and the standardized genotype.

        To avoid test data leakage, it is important to initialize the scaler
        without the test samples. Ensure that dataset splitting is done using
        separate backends or that scalers are re-computed if needed.

        """
        geno_dosage = self.backend[idx]

        if self.scaler is not None:
            return BatchDosageStandardized(
                geno_dosage,
                self.scaler.standardize_tensor(geno_dosage)
            )
        else:
            return BatchDosage(geno_dosage)

    def __len__(self) -> int:
        return len(self.backend)


class UnionGeneticDataset(Dataset):
    def __init__(
        self, dataset1: GeneticDataset, dataset2: GeneticDataset
    ):
        """Genetic dataset that combines two compatible datasets.

        Here, combination is variant-wise / column-wise.

        Both datasets will be checked for length, but not for sample
        permutation. This is left to the user.

        Both datasets should also be configured so that they generate
        compatible fields/information. For example, variant standardization
        and similar functionality should be enabled or disabled for both
        datasets.

        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        assert len(dataset1) == len(dataset2)

    def __getitem__(self, index: int) -> Batch:
        batch1 = self.dataset1[index]
        batch2 = self.dataset1[index]

        assert batch1._fields == batch2._fields

        concatenated = []
        for value1, value2 in zip(batch1, batch2):
            concatenated.append(torch.hstack((value1, value2)))

        return type(batch1)(batch1.name, batch1._fields, *concatenated)

    def __len__(self) -> int:
        return len(self.dataset1)


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
            f"{self.first_variant_index}:{self.last_variant_index}>"
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

    def get_dataset_for_chunk_id(
        self,
        chunk_id: int,
        genotype_standardization: bool = False
    ) -> GeneticDataset:
        # For now, we use the MaskBackendWrapper, but we could consider
        # implementing a "RangeBackendWrapper" which could be more efficient
        # for large chunks.
        chunk = self.get_chunk(chunk_id)
        assert chunk.first_variant_index is not None
        assert chunk.last_variant_index is not None

        indices = torch.arange(
            chunk.first_variant_index, chunk.last_variant_index + 1
        )

        backend = MaskBackendWrapper(self.backend)
        backend.keep_variants_indices(indices)

        return GeneticDataset(backend, genotype_standardization)


class MaskBackendWrapper(GeneticDatasetBackendWrapper):
    def __init__(self, backend: GeneticDatasetBackend):
        self.backend = backend

        # If None, we keep everything.
        self.variants_keep_indices: Optional[torch.Tensor] = None
        self.samples_keep_indices: Optional[torch.Tensor] = None

    def keep_variants(self, variants: Iterable[Variant]):
        keep_indices = []
        variants_set = set(variants)
        for i, v in enumerate(self.backend.get_variants()):
            if v in variants_set:
                keep_indices.append(i)

        self.variants_keep_indices = torch.tensor(keep_indices)
        if len(self.variants_keep_indices) == 0:
            raise ValueError("No overlapping variants.")

    def keep_variants_names(self, names: Iterable[str]):
        keep_indices = []
        names_set = set(names)
        for i, v in enumerate(self.backend.get_variants()):
            if v.name in names_set:
                keep_indices.append(i)

        self.variants_keep_indices = torch.tensor(keep_indices)
        if len(self.variants_keep_indices) == 0:
            raise ValueError("No overlapping variants.")

    def keep_variants_indices(self, indices: Iterable[int]):
        if isinstance(indices, np.ndarray):
            self.variants_keep_indices = torch.from_numpy(indices)
        if isinstance(indices, torch.Tensor):
            self.variants_keep_indices = indices
        else:
            self.variants_keep_indices = torch.tensor(indices)

    def keep_samples(self, samples: Iterable[str], sort: bool = False):
        keep_indices = []
        samples_set = set(samples)
        for i, s in enumerate(self.backend.get_samples()):
            if s in samples_set:
                keep_indices.append(i)

        if sort:
            keep_indices = sorted(keep_indices)

        self.samples_keep_indices = torch.tensor(keep_indices)

    def keep_samples_indices(self, indices: Iterable[int], sort: bool = False):
        if isinstance(indices, np.ndarray):
            self.samples_keep_indices = torch.from_numpy(indices)
        if isinstance(indices, torch.Tensor):
            self.samples_keep_indices = indices
        else:
            self.samples_keep_indices = torch.tensor(indices)

        if sort:
            self.samples_keep_indices = torch.sort(
                self.samples_keep_indices
            )[0]

    def get_n_samples(self) -> int:
        if self.samples_keep_indices is None:
            return len(self.backend)
        else:
            return len(self.samples_keep_indices)

    def get_n_variants(self) -> int:
        if self.variants_keep_indices is None:
            return self.backend.get_n_variants()
        else:
            return len(self.variants_keep_indices)

    def get_samples(self) -> List[str]:
        be_samples = self.backend.get_samples()
        if self.samples_keep_indices is None:
            return be_samples

        return [be_samples[i] for i in self.samples_keep_indices]

    def get_variants(self) -> List[Variant]:
        be_variants = self.backend.get_variants()
        if self.variants_keep_indices is None:
            return be_variants

        return [be_variants[i] for i in self.variants_keep_indices]

    def __getitem__(self, idx: int):
        if self.samples_keep_indices is None:
            row = self.backend[idx]

        else:
            row = self.backend[self.samples_keep_indices[idx]]

        # Subset row if needed.
        if self.variants_keep_indices is not None:
            return row[self.variants_keep_indices]
        else:
            return row

    def __len__(self):
        if self.samples_keep_indices is None:
            return len(self.backend)

        else:
            return len(self.samples_keep_indices)
