#!/bin/bash
# Builds oneTBB for the macOS wheels. The Homebrew package targets the macOS version of the runner,
# so delocate would refuse to bundle it into a wheel with a lower deployment target.
# cibuildwheel runs it once per build, see [tool.cibuildwheel.macos] in pyproject.toml.
set -euo pipefail

TBB_VERSION=v2023.1.0
PREFIX=${TBB_PREFIX:-/tmp/onetbb-install}

export MACOSX_DEPLOYMENT_TARGET=11.0
git clone --depth 1 --branch ${TBB_VERSION} https://github.com/uxlfoundation/oneTBB.git /tmp/onetbb
cmake -S /tmp/onetbb -B /tmp/onetbb/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DTBB_TEST=OFF \
    -DTBB_EXAMPLES=OFF
cmake --build /tmp/onetbb/build --parallel "$(sysctl -n hw.ncpu)"
cmake --install /tmp/onetbb/build
rm -rf /tmp/onetbb
