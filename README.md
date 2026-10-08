# PhotonTracer

[![DOI (Article)](<https://img.shields.io/badge/DOI%20(Article)-10.1016%2Fj.jqsrt.2026.109894-blue>)](https://doi.org/10.1016/j.jqsrt.2026.109894)
[![DOI (Code)](<https://img.shields.io/badge/DOI%20(Code)-10.5281%2Fzenodo.19048554-blue>)](https://doi.org/10.5281/zenodo.19048554)
![GitHub License](https://img.shields.io/github/license/unibe-icelab/photontracer)

PhotonTracer is a GPU-accelerated raytracing simulation for arbitrary media in the geometric optics limit. The geometry is represented by a mesh. It can handle media consisting of billions of arbitrary particles and simulate the interaction of millions of photons with the medium. A single photon can scatter millions of times in the medium.
The material properties are defined by a set of complex refractive indices. Up to 16 different materials can be defined in a single simulation. The simulation is written in C++ and uses the OptiX raytracing engine to accelerate the simulation on NVIDIA GPUs. A CPU backend based on Embree runs the same physics without a GPU.
The Simulation has pybind11 Python bindings and is compiled as a Python package, using scikit-build.
Therefore the full simulation, including the generation of the medium, is set up from Python and the outputs are read out as NumPy arrays.

## License

PhotonTracer is released under the BSD 3-Clause License. See the LICENSE file for more information.

## Citation

If you use this software, please cite this article:

R. Ottersberg, A. Pommerol, N. Thomas. PhotonTracer: A GPU-accelerated ray tracing simulation of light transport in highly multiple scattering media. Journal of Quantitative Spectroscopy and Radiative Transfer: 109894, 2026. [10.1016/j.jqsrt.2026.109894](https://doi.org/10.1016/j.jqsrt.2026.109894)

## Hardware Requirements

For the OptiX backend:

- NVIDIA GPU with CUDA support (compute capability 5.0 or higher)
- To make use of the accelerated ray-intersection, an GPU with RT cores is required (e.g. NVIDIA RTX series)
- NVIDIA display driver 535+ (check with nvidia-smi).

The Embree backend runs on any x86-64 CPU and needs no GPU.

## Installation

The following build tools are required to compile photontracer:

- C++ compiler (GCC 11+ on Linux, MSVC on Windows)
- Python 3 with development headers
- CMake (>=3.27)
- NVIDIA CUDA toolkit (>=12.4), for the OptiX backend
- Intel oneTBB (>=2021), for the Embree backend

And to build and run the unit tests:

- GTest

On Linux, conda-forge can be used to install all these dependencies (see the environment_linux/ file):

```bash
conda env create -f environment_linux.yml -n photontracer
conda activate photontracer
```

On Windows, the Visual Studio installer can be used to install the C++ build tools and CMake. The CUDA toolkit can be downloaded from NVIDIA.

The NVIDIA OptiX SDK is required to compile the simulation. It is bundeled as a submodule. The used version is OptiX 8.0.0, which needs a minimum display driver version of 535.

Embree 4 is bundled as a submodule and built together with photontracer.

### Clone the repository in a directory of choice including submodules

```bash
git clone --recurse-submodules https://github.com/unibe-icelab/photontracer.git
cd photontracer
```

### Build and install the Python package

Activate your virtual Python environment. For example the one created with the env with conda-forge/Miniforge3.

Build and install photontracer with scikit-build using pip:

```bash
pip install .
```

Both backends are built by default. The CMake options `PHOTONTRACER_BUILD_OPTIX` and `PHOTONTRACER_BUILD_EMBREE` switch them off. For a machine without CUDA, build the CPU backend only:

```bash
pip install . -C cmake.define.PHOTONTRACER_BUILD_OPTIX=OFF
```

### Backends

`Simulation()` uses OptiX if the build contains it and Embree otherwise. Choose one explicitly with the `backend` argument, and list those of the build with `available_backends()`:

```python
sim = photontracer.Simulation(backend=photontracer.Backend.EMBREE, cpu_threads=8)
```

`cpu_threads=0` (the default) uses all cores. The Embree backend supports only the PCG32 generator so far, and not yet the outputs `LOGS`, `LOG_OFFSETS` and `DIRECTION_HISTOGRAM_HEALPIX`. Instances can be nested up to the CMake setting `PHOTONTRACER_EMBREE_INSTANCE_LEVELS` (default 4); `Simulation.max_nested_geometry_levels` is limited to that plus one.

### Random number generator

The kernels use PCG32 by default. It is much cheaper to seed than curand's MRG32k3a, which the 1.0
releases used, and for scenes with little physics per ray this is most of the run time. To use
curand again, for example to reproduce results of a 1.0 release ray by ray, set it on the simulation:

```python
sim.random_number_generator = photontracer.RandomNumberGenerator.MRG32K3A
```

Both generators give the same results statistically, but not ray by ray, so a seed only reproduces
a run with the same generator (and the same build).

## Usage

Refer to the example jupyter notebooks in `examples/` and the docstrings of the Python objects.

## Tests

Unit test on the light scattering kernels are written in C++ using the GTest framework. They are located in `src/tests`. To run the tests, build the project with CMake, with the option `PHOTONTRACER_BUILD_TESTS` set to `ON`.

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

With both backends off no Python module is built, since there is no raytracing backend to run it with. The Python tests run on every backend of the build.

To test the Python module, complete integration tests are available in the `photontracer/tests` directory. They can be run with pytest from the root directory:

```bash
pytest photontracer/test
```
