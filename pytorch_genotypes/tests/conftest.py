

import os
import csv
import itertools

import pytest
import numpy as np
from pkg_resources import resource_filename

from ..dataset import NumpyBackend


def get_small_backend():
    filename = resource_filename(
        __name__,
        os.path.join("test_data", "1kg_common_norel_thinned25.pkl")
    )

    backend = NumpyBackend.load(filename)

    # The backend was created with lazy_pickle = True meaning that we need
    # to read the numpy matrix ourselves.
    backend.m = np.load(
        resource_filename(
            __name__,
            os.path.join("test_data", "1kg_common_norel_thinned25.npz")
        )
    )["arr_0"]

    return backend


@pytest.fixture
def small_backend():
    return get_small_backend()


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
