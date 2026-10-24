import pytest

from photontracer import Backend, available_backends, is_cuda_available


def _usable(backend):
    # A build with OptiX also runs on a machine without an NVIDIA GPU or driver
    return backend != Backend.OPTIX or is_cuda_available()


@pytest.fixture(
    params=[
        pytest.param(b, id=b.name, marks=[] if _usable(b) else pytest.mark.skip(reason="no CUDA device"))
        for b in available_backends()
    ]
)
def backend(request):
    return request.param


@pytest.fixture
def backends():
    """The backends that can run here, for tests that compare them."""
    return [b for b in available_backends() if _usable(b)]
