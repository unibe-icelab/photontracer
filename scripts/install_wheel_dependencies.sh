#!/bin/bash
# Installs the build dependencies of the wheels in the manylinux_2_28 image (AlmaLinux 8).
# cibuildwheel runs it once per container, see [tool.cibuildwheel.linux] in pyproject.toml.
set -euo pipefail

CUDA_VERSION=12-4
TBB_VERSION=v2023.1.0

# CUDA 12.4 supports GCC up to 13; nvcc uses it as host compiler through CUDAHOSTCXX
dnf install -y dnf-plugins-core gcc-toolset-13-gcc-c++

dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y \
    cuda-nvcc-${CUDA_VERSION} \
    cuda-cudart-devel-${CUDA_VERSION} \
    cuda-driver-devel-${CUDA_VERSION} \
    cuda-cccl-${CUDA_VERSION} \
    libcurand-devel-${CUDA_VERSION}

# AlmaLinux 8 has TBB 2018, Embree needs oneTBB 2021 or newer
git clone --depth 1 --branch ${TBB_VERSION} https://github.com/uxlfoundation/oneTBB.git /tmp/onetbb
cmake -S /tmp/onetbb -B /tmp/onetbb/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DTBB_TEST=OFF \
    -DTBB_EXAMPLES=OFF
cmake --build /tmp/onetbb/build --parallel "$(nproc)"
cmake --install /tmp/onetbb/build
rm -rf /tmp/onetbb

# auditwheel finds libtbb through the loader cache to bundle it
echo /usr/local/lib > /etc/ld.so.conf.d/usr-local.conf
ldconfig
