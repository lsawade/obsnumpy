import pytest

from obsnumpy.pipeline import cupy_or_none


@pytest.fixture(scope="session")
def cp():
    """The cupy module; the test is skipped without CuPy or a CUDA device."""
    cupy = cupy_or_none()
    if cupy is None:
        pytest.skip("CuPy or a CUDA device is not available")
    return cupy
