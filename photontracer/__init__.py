"""Photon Tracer - GPU-accelerated ray tracing simulation"""
from importlib.metadata import PackageNotFoundError, version

# Import and re-export the public API from the C++ extension module.
try:
    from .photontracer_bindings import (
        is_cuda_available,
        GeometryType,
        MESH,
        MESH_INSTANCED,
        LengthUnit,
        MICRO_METER,
        MILLI_METER,
        METER,
        MaterialType,
        DIFFUSE,
        REFRACTIVE,
        REFLECTIVE,
        VOLUME_SCATTERING,
        OutputType,
        LAST_DIRECTION,
        LAST_POSITION,
        RAY_STATE,
        LAST_MEDIUM_ID,
        SCATTERING_COUNT,
        NUMBER_OF_WARNINGS,
        STOKES_VECTOR,
        STOKES_VECTOR_IN,
        OPTICAL_PATH_LENGTH,
        SOURCE_DIRECTION,
        SOURCE_POSITION,
        SCATTERING_ANGLE,
        Q_MINUS_AXIS_IN,
        LOGS,
        LOG_OFFSETS,
        DIRECTION_HISTOGRAM_HEALPIX,
        RefractiveIndex,
        MaterialProperties,
        Material,
        Simulation,
        IGeometry,
        MeshGeometry,
        InstanceGeometry,
        IRayGenerator,
        ParallelRayGenerator,
        IsotropicRayGenerator,
        CameraRayGenerator,
    )
    assert is_cuda_available(), "CUDA is not available. Please ensure that CUDA is installed and a compatible GPU is present."
except ImportError as e:
    error_msg = (
        "Failed to import photontracer\n\n"
    )
    error_msg += f"\nOriginal error: {str(e)}"
    raise ImportError(error_msg) from e


# Define the export surface for ``from photontracer import *``.
__all__ = [
    'is_cuda_available',
    'GeometryType',
    'MESH',
    'MESH_INSTANCED',
    'LengthUnit',
    'MICRO_METER',
    'MILLI_METER',
    'METER',
    'MaterialType',
    'DIFFUSE',
    'VOLUME_SCATTERING',
    'REFRACTIVE',
    'REFLECTIVE',
    'OutputType',
    'LAST_DIRECTION',
    'LAST_POSITION',
    'RAY_STATE',
    'LAST_MEDIUM_ID',
    'SCATTERING_COUNT',
    'NUMBER_OF_WARNINGS',
    'STOKES_VECTOR',
    'STOKES_VECTOR_IN',
    'OPTICAL_PATH_LENGTH',
    'SOURCE_DIRECTION',
    'SOURCE_POSITION',
    'SCATTERING_ANGLE',
    'Q_MINUS_AXIS_IN',
    'LOGS',
    'LOG_OFFSETS',
    'DIRECTION_HISTOGRAM_HEALPIX',
    'RefractiveIndex',
    'MaterialProperties',
    'Material',
    'Simulation',
    'IGeometry',
    'MeshGeometry',
    'InstanceGeometry',
    'IRayGenerator',
    'ParallelRayGenerator',
    'IsotropicRayGenerator',
    'CameraRayGenerator',
]

# Patch Simulation.run() to flush stdout before running the simulation
# This ensures that jupyter notebooks display that the simulation is running
_original_run = Simulation.run


def _patched_run(self, *args, **kwargs):
    import sys
    print("Starting simulation")
    sys.stdout.flush()
    return _original_run(self, *args, **kwargs)


Simulation.run = _patched_run

try:
    __version__ = version("photontracer")
except PackageNotFoundError:
    raise ImportError("photontracer package not found.")
