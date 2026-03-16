# PhotonTracer

PhotonTracer is a GPU-accelerated raytracing simulation for arbitrary media in the geometric optics limit. The geometry is represented by a mesh. It can handle media consisting of billions of arbitrary particles and simulate the interaction of millions of photons with the medium. A single photon can scatter millions of times in the medium.
The material properties are defined by a set of complex refractive indices. Up to 16 different materials can be defined in a single simulation. The simulation is written in C++ and uses the OptiX raytracing engine to accelerate the simulation on NVIDIA GPUs.
The Simulation has pybind11 Python bindings and is compiled as a Python package, using scikit-build.
Therefore the full simulation, including the generation of the medium, is set up from Python and the outputs are read out as NumPy arrays.

## License

PhotonTracer is released under the BSD 3-Clause License. See the LICENSE file for more information.

## Citation

If you use this software, please cite this article:
R. Ottersberg, A. Pommerol, N. Thomas. PhotonTracer: A GPU-accelerated ray tracing simulation of light transport in highly multiple scattering media. Journal of Quantitative Spectroscopy and Radiative Transfer: 109894, 2026. [10.1016/j.jqsrt.2026.109894](https://doi.org/10.1016/j.jqsrt.2026.109894)

## Hardware Requirements

- NVIDIA GPU with CUDA support (compute capability 5.0 or higher)
- To make use of the accelerated ray-intersection, an GPU with RT cores is required (e.g. NVIDIA RTX series)
- NVIDIA display driver 535+ (check with nvidia-smi).

## Installation

The following build tools are required to compile photontracer:

- C++ compiler (GCC 11+ on Linux, MSVC on Windows)
- Python 3 with development headers
- CMake (>=3.27)
- NVIDIA CUDA toolkit (>=12.4)

And to build and run the unit tests:

- GTest

On Linux, conda-forge can be used to install all these dependencies (see the environment_linux/ file):

```bash
conda env create -f environment_linux.yml -n photontracer
conda activate photontracer
```

On Windows, the Visual Studio installer can be used to install the C++ build tools and CMake. The CUDA toolkit can be downloaded from NVIDIA.

The NVIDIA OptiX SDK is required to compile the simulation. It is bundeled as a submodule. The used version is OptiX 8.0.0, which needs a minimum display driver version of 535.

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

To test the Python module, complete integration tests are available in the `photontracer/tests` directory. They can be run with pytest from the root directory:

```bash
pytest photontracer/test
```
