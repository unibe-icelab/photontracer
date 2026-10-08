import pytest

from photontracer import available_backends


@pytest.fixture(params=available_backends(), ids=lambda backend: backend.name)
def backend(request):
    return request.param
