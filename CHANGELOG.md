# PhotonTracer Changelog

All notable changes to this project will be documented in this file.

## 1.0.x - Unreleased

- Add Changelog file
- Add CI workflow that compiles the OptiX build without a GPU
- Run the CI checks for PRs and pushes to the dev branch
- Add `benchmarks/benchmark.py` for timing synthetic scenes and comparing against a stored baseline
- Fix crash when a `Simulation` is used after another one has been destroyed (the destructor reset the whole CUDA device)
- Reject more than 16 materials, instance material IDs outside the configured materials, and mesh scenes with fewer than 2 materials; before, these silently absorbed rays or wrote past a fixed-size array
- Fix the debug-build bounds checks, which rejected the valid material ID 15
- Move the OptiX context and pipelines from `Simulation` into an OptiX raytracing backend behind a new `IRaytracingBackend` interface; the Python API and the results are unchanged
- Add `benchmarks/golden.py` to check that two builds produce identical output
- Make the physics (`light_scattering`, `complex_f`, material interaction, ray generation) host-compilable in portable headers, and add CPU-only tests (`runHostTests`) that run without a GPU
- Allocate and copy the output buffers through the raytracing backend instead of calling CUDA from `Simulation`
- Move the OptiX acceleration structures out of the geometry classes into the OptiX backend, which also fixes a crash when two simulations share one geometry
- Add the CMake option `PHOTONTRACER_BUILD_OPTIX` (default on); with it off, CUDA is not needed and only the CPU tests are built. A CI job runs them without a GPU
- Use PCG32 as the default random number generator of the OptiX kernels. It is much cheaper to seed than curand's MRG32k3a, which makes scenes with little physics per ray far faster and heavy ones about 15% faster. Results agree statistically with the 1.0 releases but not ray by ray. `Simulation.random_number_generator` selects between `PCG32` and `MRG32K3A` (the curand generator of the 1.0 releases) at run time
- Add statistical tests of the generator (Fresnel reflectance, diffuse albedo, uniformity of the isotropic source, seed reproducibility), and `--summary` / `--compare-summary` in `benchmarks/golden.py` to compare the statistics of two builds
- Add an Embree backend that runs the simulation on the CPU with TBB, selected with the `backend` argument of `Simulation`. The new CMake option `PHOTONTRACER_BUILD_EMBREE` (default on) builds it, and `PHOTONTRACER_BUILD_OPTIX=OFF` builds the Python module without CUDA. `available_backends()` lists the backends of a build. It supports instances, nested up to `PHOTONTRACER_EMBREE_INSTANCE_LEVELS` levels (default 4). The AVX512 kernels of Embree are not built, to keep the build time down
- Compact mesh acceleration structures after building, controlled by the new `compact` argument of `MeshGeometry` (default on)

## 1.0.2 - 18-03-2026

- Update citation information (ISSN and volume of the journal article)

## 1.0.1 - 16-03-2026

- Add citation information and update tests to match API changes

## 1.0.0 - 12-03-2026

- Initial release of PhotonTracer
