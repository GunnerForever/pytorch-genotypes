import numpy as np
from ..dataset import NumpyBackend, ZarrBackend


def test_generic_range_extraction(
    small_np_backend: NumpyBackend,
    caplog
):
    assert small_np_backend.m is not None

    np.testing.assert_array_equal(
        small_np_backend.m[:, 1:3],
        super(type(small_np_backend), small_np_backend).extract_range(1, 2)
    )

    assert len(caplog.records) == 1


def test_np_range_extraction(
    small_np_backend: NumpyBackend,
    caplog
):
    assert small_np_backend.m is not None

    np.testing.assert_array_equal(
        small_np_backend.m[:, 1:3],
        small_np_backend.extract_range(1, 2)
    )

    assert len(caplog.records) == 0


def test_zarr_range_extraction(
    small_zarr_backend: ZarrBackend,
    small_np_backend: NumpyBackend,
    caplog
):
    assert small_np_backend.m is not None

    np.testing.assert_array_equal(
        small_np_backend.m[:, 1:3],
        small_zarr_backend.extract_range(1, 2)
    )

    assert len(caplog.records) == 0
