"""Types of the arguments of the bindings that have no Python class of their own."""
import typing

import numpy
from numpy.typing import NDArray

Vector3 = typing.Union[NDArray[numpy.floating], NDArray[numpy.integer], typing.Tuple[float, float, float], typing.List[float]]
"""A vector of 3 numbers: a NumPy array of 3 elements (cast to ``float32``), a 3-tuple or a list of 3 numbers."""

Vector4 = typing.Union[NDArray[numpy.floating], NDArray[numpy.integer], typing.Tuple[float, float, float, float], typing.List[float]]
"""A vector of 4 numbers: a NumPy array of 4 elements (cast to ``float32``), a 4-tuple or a list of 4 numbers."""
