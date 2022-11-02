# flake8: noqa
from typing import Dict, Type

from .core import (FixedSizeChunks, GeneticDataset, GeneticDatasetBackend,
                   MaskBackendWrapper)
from .numpy_backend import NumpyBackend
from .phenotype_dataset import PhenotypeGeneticDataset
from .zarr_backend import ZarrBackend


BACKENDS: Dict[str, Type[GeneticDatasetBackend]] = {
    "ZarrBackend": ZarrBackend,
    "NumpyBackend": NumpyBackend
}
