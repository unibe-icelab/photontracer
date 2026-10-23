import os

import pytest

from photontracer import available_backends

# PHOTONTRACER_TEST_BACKENDS (e.g. "EMBREE") limits the tests to some backends, for a build with
# OptiX on a machine without a GPU
_selected = os.environ.get("PHOTONTRACER_TEST_BACKENDS")
_backends = [b for b in available_backends() if _selected is None or b.name in _selected.split(",")]


@pytest.fixture(params=_backends, ids=lambda backend: backend.name)
def backend(request):
    return request.param
