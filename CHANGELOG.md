# PhotonTracer Changelog

All notable changes to this project will be documented in this file.

## 1.1.0 - Unreleased

### Breaking changes

- Results agree with the 1.0 releases statistically but not ray by ray: the OptiX kernels use PCG32 instead of curand's MRG32k3a by default, which makes scenes with little physics per ray far faster and heavy ones about 15% faster, and `DIFFUSE` surfaces sample their directions with Malley's method
- Require Python 3.9 or newer, and declare `numpy` as a dependency
- Remove the `LOGS` and `LOG_OFFSETS` outputs and the CMake option `ENABLE_DEBUG_PRINT_KERNEL`; the new ray trace replaces them
- Print progress messages (device in use, compile and launch times) only after `photontracer.set_verbose(True)`
- Reject more than 16 materials, instance material IDs outside the configured materials, and mesh scenes with fewer than 2 materials; before, these silently absorbed rays or wrote past a fixed-size array
- Invalid values for `Simulation.stokes_vector`, `max_nested_geometry_levels`, `direction_healpix_nside` and `length_unit` raise `ValueError` instead of `RuntimeError`. A vector argument with a wrong number of values raises `ValueError`, one of a wrong type a `TypeError`. `InstanceGeometry` rejects out-of-range `particle_type_ids` when it is created instead of in `run()`

### Added

- Embree backend that runs the simulation on the CPU, selected with the `backend` argument of `Simulation`; `available_backends()` lists the backends of a build. It supports meshes, instances nested up to `PHOTONTRACER_EMBREE_INSTANCE_LEVELS` levels (default 4) and the `DIRECTION_HISTOGRAM_HEALPIX` output, and only the `PCG32` generator
- `Simulation.random_number_generator` to select `PCG32` or `MRG32K3A` (curand, OptiX only) at run time
- `Simulation.trace_rays`, `max_trace_steps` and `get_trace()` to record the steps (positions, directions, media, Stokes vector, event) of up to 256 chosen rays on both backends
- `DiffuseRayGenerator`, a disk source emitting cosine-weighted directions around an axis to model diffuse illumination of a surface
- `disk_normal` argument and property of `ParallelRayGenerator`: the plane of the source disk, perpendicular to the beam by default. A disk parallel to the illuminated surface keeps the footprint of a tilted beam the same circle at every zenith angle
- `FIRST_POSITION` output: where each ray first interacted, and NaN for rays that left the scene without an interaction
- `compact` argument of `MeshGeometry` (default on) to compact the mesh acceleration structures after building
- CMake options `PHOTONTRACER_BUILD_OPTIX` and `PHOTONTRACER_BUILD_EMBREE` (both default on; without OptiX, CUDA is not needed), and `PHOTONTRACER_USE_FASTMATH` (default on), which now also compiles the Embree physics with `-ffast-math -fno-finite-math-only`
- Documentation of the Python API in `help()` and a type stub, `photontracer_bindings.pyi`. The outputs document the codes of `RAY_STATE`, the frame of the Stokes vectors, the units and what `NUMBER_OF_WARNINGS` counts. Error messages name the Python attributes instead of the C++ functions

### Fixed

- Crash when a `Simulation` is used after another one has been destroyed (the destructor reset the whole CUDA device)
- Crash when two simulations share one geometry
- Debug-build bounds checks rejecting the valid material ID 15
- `ParallelRayGenerator.direction` not being normalised when set after construction
- `Material(REFLECTIVE, reflectivity)` and `Material(REFRACTIVE, n)` with a real `n` raising `ValueError`; the reflectivity of a mirror without fuzziness is now checked to be in [0, 1]
- `calculate_volume_fraction()` discarding a pending rebuild of the ray tracing pipeline, so that e.g. a change of geometry type or of `random_number_generator` was ignored by the next `run()`
- Wrong defaults in the documentation (`seed` is 42, `use_complex_fresnel` is on, `max_nested_geometry_levels` is 5)
- The wheel containing the headers, static libraries and documentation of Embree and the OptiX Toolkit, and, when built without `--as-needed`, requiring `libcuda.so.1` to import the module

### Internal

- The OptiX context, pipelines, acceleration structures and output buffers moved from `Simulation` and the geometries into an OptiX backend behind an `IRaytracingBackend` interface
- The physics is host-compilable, with CPU-only tests (`runHostTests`) and statistical tests of the generator
- `benchmarks/benchmark.py` for timing, `benchmarks/golden.py` to compare the output or statistics of two builds
- CI compiles the OptiX build without a GPU, runs the CPU tests for Python 3.9 and 3.13, and checks that the committed stub is up to date (`scripts/generate_stubs.py`)
- A workflow builds manylinux_2_28 wheels for Python 3.9 to 3.14 with cibuildwheel (CUDA 12.4, OptiX and Embree), tests the Embree backend and publishes development versions to TestPyPI. `PHOTONTRACER_TEST_BACKENDS` limits the Python tests to some backends
- Add this changelog, and update LICENSE with the current affiliation of Rafael Ottersberg

## 1.0.2 - 18-03-2026

- Update citation information (ISSN and volume of the journal article)

## 1.0.1 - 16-03-2026

- Add citation information and update tests to match API changes

## 1.0.0 - 12-03-2026

- Initial release of PhotonTracer
