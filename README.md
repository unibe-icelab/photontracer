# PhotonTracer

[![DOI (Article)](<https://img.shields.io/badge/DOI%20(Article)-10.1016%2Fj.jqsrt.2026.109894-blue>)](https://doi.org/10.1016/j.jqsrt.2026.109894)
[![DOI (Code)](<https://img.shields.io/badge/DOI%20(Code)-10.5281%2Fzenodo.18989228-blue>)](https://doi.org/10.5281/zenodo.18989228)
![GitHub License](https://img.shields.io/github/license/unibe-icelab/photontracer)

If you use PhotonTracer in your research, please [cite it](#citation).

PhotonTracer is a GPU-accelerated ray tracing simulation of light transport in highly multiple scattering media, in the geometric optics limit. It follows millions of photons through media of billions of particles, each photon scattering up to millions of times, for example in snow, ice or regolith. The geometry is a mesh, and the materials are defined by complex refractive indices (up to 16 per simulation).

It runs on NVIDIA GPUs with OptiX, or on the CPU with Embree. The whole simulation, including the generation of the medium, is set up from Python, and the outputs are read as NumPy arrays.

## Installation

```bash
pip install photontracer
```

Wheels for Python 3.9 to 3.14 are available for:

| Platform | Requirements | Backends |
| --- | --- | --- |
| Linux (x86-64) | glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, RHEL 8+) | OptiX and Embree |
| Windows (x86-64) | | OptiX and Embree |
| macOS (Apple silicon) | macOS 11 or newer | Embree only, there is no CUDA on the Mac |

The OptiX backend needs an NVIDIA GPU and a current display driver (535 or newer on Linux, check with `nvidia-smi`). It loads the driver at run time, so CUDA does not have to be installed. Compute capability 5.0 or higher is supported, and a GPU with RT cores (the NVIDIA RTX series) makes the ray intersection faster. Without a CUDA device, `Simulation()` uses the Embree backend on the CPU and emits a warning.

For other platforms, see [building from source](#build-from-source).

## Quick start

```python
import numpy as np
import photontracer as phtr

# A cube of 100 units, made of 12 triangles
vertices = np.array([[x, y, z] for z in (0, 100) for y in (-50, 50) for x in (-50, 50)], dtype=np.float32)
faces = np.array([
    [0, 2, 1], [0, 3, 2], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
    [2, 7, 3], [2, 6, 7], [1, 5, 6], [1, 6, 2], [3, 7, 4], [3, 4, 0],
], dtype=np.uint32)

sim = phtr.Simulation()
sim.geometry = phtr.MeshGeometry(vertices, faces)
sim.materials = [
    phtr.Material(phtr.MaterialType.REFRACTIVE, 1 + 0j),         # outside: vacuum
    phtr.Material(phtr.MaterialType.REFRACTIVE, 1.33 + 1e-6j),   # inside: water
]
sim.wavelength_um = 0.6
sim.ray_generator = phtr.ParallelRayGenerator(
    number_of_rays=100_000,
    origin=np.array([0, 0, 200]),
    direction=np.array([0, 0, -1]),
    offset_radius=40,
)
sim.outputs = [phtr.OutputType.SCATTERING_COUNT]

sim.run()
print(sim.get_output_buffer(phtr.OutputType.SCATTERING_COUNT).mean())
```

[`examples/minimal.py`](examples/minimal.py) is the shortest script, and the [example notebook](examples/ice_particle_reflectance_spectrum.ipynb) shows a full workflow. The examples and snippets here import the package as `phtr`, so that typing `phtr.` offers every class and function of the package in an IDE. The API is documented in the docstrings, so `help(phtr.Simulation)` shows it.

## Usage

### Backends

`Simulation()` uses OptiX if the build contains it and a CUDA device is available, and Embree otherwise. Falling back to Embree in a build with OptiX, e.g. a wheel from PyPI on a machine without an NVIDIA GPU, emits a `RuntimeWarning`. Choose one explicitly with the `backend` argument, and list those of the build with `available_backends()`:

```python
sim = phtr.Simulation(backend=phtr.Backend.EMBREE, cpu_threads=8)
```

`cpu_threads=0` (the default) uses all cores. The Embree backend supports only the PCG32 generator. `Simulation.max_nested_geometry_levels` is limited by the nesting depth the build was compiled for (4 instance levels by default, so at most 5 with the mesh).

### Random number generator

The kernels use PCG32 by default. To use curand's MRG32k3a instead (OptiX only), set it on the simulation:

```python
sim.random_number_generator = phtr.RandomNumberGenerator.MRG32K3A
```

The generators agree statistically but not ray by ray, so a seed only reproduces a run with the same generator and build.

### Output messages

The library is quiet by default. To see which device is used and how long compiling and launching take, call `phtr.set_verbose(True)`.

### Tracing single rays

To see what happens to individual rays, list their indices (the position of the ray in the output buffers) and read the steps after the run:

```python
sim.trace_rays = [12, 40517]   # at most 256 rays
sim.max_trace_steps = 1000     # steps stored per ray; later ones are only counted
sim.run()
steps = sim.get_trace()        # structured numpy array, ordered by ray and step
steps[["ray", "step", "event", "origin_in", "direction_out"]]
```

A step goes from the start of the ray, or the last interaction, to the next one. It holds the ray before and after (position, direction, medium), the Stokes vector, the optical path length and a `TraceEvent` (interaction, escaped, absorbed, ...). Seeds are deterministic, so a ray that stands out in a normal run, for example one with a very high `SCATTERING_COUNT`, can be traced by its index in the next run.

## Citation

If you use this software, please cite this article:

R. Ottersberg, A. Pommerol, N. Thomas. PhotonTracer: A GPU-accelerated ray tracing simulation of light transport in highly multiple scattering media. Journal of Quantitative Spectroscopy and Radiative Transfer 357: 109894, 2026. [10.1016/j.jqsrt.2026.109894](https://doi.org/10.1016/j.jqsrt.2026.109894)

```bibtex
@article{ottersberg2026photontracer,
  author  = {Ottersberg, R. and Pommerol, A. and Thomas, N.},
  title   = {PhotonTracer: A GPU-accelerated ray tracing simulation of light transport in highly multiple scattering media},
  journal = {Journal of Quantitative Spectroscopy and Radiative Transfer},
  volume  = {357},
  pages   = {109894},
  year    = {2026},
  doi     = {10.1016/j.jqsrt.2026.109894}
}
```

## Build from source

The following build tools are required to compile photontracer:

- C++ compiler (GCC 11+ on Linux, MSVC on Windows, Apple Clang on macOS)
- Python 3.9 or newer with development headers
- CMake (>=3.27)
- NVIDIA CUDA toolkit, for the OptiX backend (12.4 is tested, CMake requires at least 12.0)
- Intel oneTBB (>=2021), for the Embree backend

And to build and run the unit tests:

- GTest

### Linux

conda-forge can be used to install all these dependencies (see `environment_linux.yml`):

```bash
conda env create -f environment_linux.yml -n photontracer
conda activate photontracer
```

### Windows

The Visual Studio installer can be used to install the C++ build tools and CMake. The CUDA toolkit can be downloaded from NVIDIA.

### macOS

There is no CUDA on the Mac, so only the Embree backend is built; the OptiX backend is off by default there.

```bash
brew install cmake tbb
```

### Submodules

The NVIDIA OptiX SDK is required to compile the simulation. It is bundled as a submodule. The used version is OptiX 8.0.0, which needs a minimum display driver version of 535.

Embree 4 is bundled as a submodule and built together with photontracer.

### Clone the repository including submodules

```bash
git clone --recurse-submodules https://github.com/unibe-icelab/photontracer.git
cd photontracer
```

### Build and install the Python package

Activate your virtual Python environment, for example the one created with conda-forge/Miniforge3, then build and install photontracer with scikit-build-core using pip:

```bash
pip install .
```

Both backends are built by default, except on macOS. The CMake options `PHOTONTRACER_BUILD_OPTIX` and `PHOTONTRACER_BUILD_EMBREE` switch them off. For a machine without CUDA, build the CPU backend only:

```bash
pip install . -C cmake.define.PHOTONTRACER_BUILD_OPTIX=OFF
```

The Embree backend supports instances nested up to the CMake setting `PHOTONTRACER_EMBREE_INSTANCE_LEVELS` (default 4).

### Fast math

By default the CUDA kernels are compiled with `--use_fast_math`, as in the 1.0 releases, and the physics of the Embree backend with `-ffast-math -fno-finite-math-only`. The second flag has to stay: NaN encodes the random polarization and undefined directions, and plain `-ffast-math` would make `isnan()` always false. For standard IEEE arithmetic switch it off:

```bash
pip install . -C cmake.define.PHOTONTRACER_USE_FASTMATH=OFF
```

On an RTX GPU fast math is about 3–7% faster; on the CPU the difference is within the noise. Results differ in the last bits and agree statistically.

## Tests

Unit tests of the light scattering kernels are written in C++ using the GTest framework. They are located in `src/tests`. To run the tests, build the project with CMake, with the option `PHOTONTRACER_BUILD_TESTS` set to `ON`:

```bash
mkdir build
cd build
cmake -DPHOTONTRACER_BUILD_TESTS=ON ..
cmake --build .
```

After building, run the tests in the build directory with:

```bash
ctest
```

The tests of the physics (`runHostTests`) also run on the CPU. They need neither CUDA nor a GPU, and can be built on their own with:

```bash
cmake -DPHOTONTRACER_BUILD_OPTIX=OFF -DPHOTONTRACER_BUILD_EMBREE=OFF -DPHOTONTRACER_BUILD_TESTS=ON ..
```

With both backends off no Python module is built, since there is no raytracing backend to run it with.

The integration tests of the Python module are in `photontracer/test`. They run on every backend of the build and can be run with pytest from the root directory:

```bash
pytest photontracer/test
```

## Python documentation

The Python API is documented in the bindings (`src/photonTracer/photontracer_bindings.cpp`, and the longer texts in `photontracer_bindings_doc.h`), so `help()` shows it. The type stub `photontracer/photontracer_bindings.pyi` that IDEs read is generated from them; do not edit it by hand. After changing the bindings, install the package and regenerate the stub:

```bash
pip install pybind11-stubgen==3.0.0
python scripts/generate_stubs.py
```

CI checks that the committed stub is up to date with `python scripts/generate_stubs.py --check`.

The changes of each release are listed in the [changelog](CHANGELOG.md).

## License

PhotonTracer is released under the BSD 3-Clause License. See the LICENSE file for more information.
