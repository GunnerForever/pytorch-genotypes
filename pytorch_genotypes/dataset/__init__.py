# flake8: noqa
from .core import GeneticDataset, GeneticDatasetBackend, FixedSizeChunks
from .zarr_backend import ZarrBackend
from .numpy_backend import NumpyBackend
from .phenotype_dataset import PhenotypeGeneticDataset

from typing import Dict

BACKENDS: Dict[str, GeneticDatasetBackend] = {
    "ZarrBackend": ZarrBackend,
    "NumpyBackend": NumpyBackend
}
