// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <vector>
#include <cstdint>
#include <optix.h>
#include <cuda_runtime.h>


int buildGasFromMesh(
    const std::vector<float3> &meshVertices,
    const std::vector<unsigned int> &meshIndices,
    OptixDeviceContext &context,
    OptixTraversableHandle &gasHandle, CUdeviceptr &dGasOutputBuffer,
    bool compact = true);

/// Builds an instance acceleration structure over the given instances
void buildIasFromInstances(
    const std::vector<OptixInstance> &instances,
    OptixDeviceContext &context,
    OptixTraversableHandle &iasHandle, CUdeviceptr &dIasOutputBuffer);
