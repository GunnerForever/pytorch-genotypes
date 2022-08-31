
import numpy as np

from ..dataset import FixedSizeChunks


def test_chunk_count(small_np_backend):
    # There are 25 variants in the small backend.
    # The file "expected_chunks_curated.csv" shows the expected allocations.

    chunks = FixedSizeChunks(small_np_backend, max_variants_per_chunk=3)
    assert len(chunks) == 16


def test_chunk_size_3(small_np_backend, chunks_k3_truth):
    # There are 25 variants in the small backend.
    # The file "expected_chunks_curated.csv" shows the expected allocations.

    chunks = FixedSizeChunks(small_np_backend, max_variants_per_chunk=3)

    expected = chunks_k3_truth

    for observed, expected in zip(chunks.chunks, expected):
        assert observed.first_variant_index == expected[0]
        assert observed.last_variant_index == expected[1]


def test_chunk_extract_tensor(small_np_backend, chunks_k3_truth):
    chunks = FixedSizeChunks(small_np_backend, max_variants_per_chunk=3)
    t = chunks.get_tensor_for_chunk_id(5)

    l, r = chunks_k3_truth[5]
    expected = small_np_backend.m[:, l:(r+1)]

    np.testing.assert_array_equal(t.numpy(), expected)
