

import os
import csv
import tempfile
import itertools

import pytest
import torch
import numpy as np
from pkg_resources import resource_filename

from ..dataset import NumpyBackend, ZarrBackend


try:
    import geneparse
    GENEPARSE_AVAIL = True
except ImportError:
    GENEPARSE_AVAIL = False


def get_small_np_backend():
    filename = resource_filename(
        __name__,
        os.path.join("test_data", "1kg_common_norel_thinned25.pkl")
    )

    backend = NumpyBackend.load(filename)

    # The backend was created with lazy_pickle = True meaning that we need
    # to read the numpy matrix ourselves.
    backend.m = torch.from_numpy(np.load(
        resource_filename(
            __name__,
            os.path.join("test_data", "1kg_common_norel_thinned25.npz")
        )
    )["arr_0"])

    return backend


def create_and_get_small_zarr_backend():
    if not GENEPARSE_AVAIL:
        raise RuntimeError("geneparse required to create Zarr backend.")

    plink_prefix = resource_filename(
        __name__,
        os.path.join("test_data", "1kg_common_norel_thinned25.bed")
    )[:-4]

    reader = geneparse.parsers["plink"](plink_prefix)

    tmp_filename = tempfile.NamedTemporaryFile()
    backend = ZarrBackend(reader, prefix=tmp_filename.name)

    return backend


@pytest.fixture
def small_np_backend() -> NumpyBackend:
    backend = get_small_np_backend()
    assert backend.m is not None
    return backend


@pytest.fixture
def small_zarr_backend() -> ZarrBackend:
    return create_and_get_small_zarr_backend()


@pytest.fixture
def chunks_k3_truth():
    filename = resource_filename(
        __name__,
        os.path.join("test_data", "expected_chunks_curated_k3.csv")
    )

    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        # Tuples of (chunk_id, variant_index)
        variant_allocations = [(int(row[0]), int(row[1])) for row in reader]

    # Recreate the chunks.
    chunks = []
    for _, rows in itertools.groupby(variant_allocations,
                                     key=lambda tu: tu[0]):
        rows = list(rows)
        first = rows[0][1]
        last = rows[-1][1]
        chunks.append((first, last))

    return chunks
