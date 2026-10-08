// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <vector>
#include <cstdint>
#include <optix.h>
#include <cuda_runtime.h>


int buildGasFromMesh(
    std::vector<float3> &meshVertices,
    std::vector<uint32_t> &meshIndices,
    OptixDeviceContext &context,
    OptixTraversableHandle &gasHandle, CUdeviceptr &dGasOutputBuffer);