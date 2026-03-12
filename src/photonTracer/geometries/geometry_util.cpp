// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

// Portions of this code were derived from NVIDIA OptiX sample code.
// See THIRD_PARTY_NOTICES.md for full license text.

#include <vector>
#include <chrono>
#include <cassert>

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>


int buildGasFromMesh(
    std::vector<float3> &meshVertices,
    std::vector<uint32_t> &meshIndices,
    OptixDeviceContext &context,
    OptixTraversableHandle &gasHandle, CUdeviceptr &dGasOutputBuffer)
{

    uint32_t maxPrimitivesPerGas = 0;
    optixDeviceContextGetProperty(
        context,
        OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
        &maxPrimitivesPerGas,
        sizeof(maxPrimitivesPerGas));

    assert(meshIndices.size() % 3 == 0 && "Index count must be a multiple of 3");

    assert(meshIndices.size() / 3 <= maxPrimitivesPerGas &&
           ("Mesh has too many triangles for one Geometry Acceleration Structure on this device which is: " + std::to_string(maxPrimitivesPerGas)).c_str());

    size_t verticesSize = meshVertices.size() * sizeof(float3);
    size_t indicesSize = meshIndices.size() * sizeof(uint32_t);

    CUdeviceptr dVertices = 0;
    OTK_ERROR_CHECK(cudaMalloc(reinterpret_cast<void **>(&dVertices), verticesSize));
    OTK_ERROR_CHECK(cudaMemcpy(reinterpret_cast<void *>(dVertices),
                               meshVertices.data(), verticesSize,
                               cudaMemcpyHostToDevice));

    CUdeviceptr dIndices = 0;
    OTK_ERROR_CHECK(cudaMalloc(reinterpret_cast<void **>(&dIndices), indicesSize));
    OTK_ERROR_CHECK(cudaMemcpy(reinterpret_cast<void *>(dIndices),
                               meshIndices.data(), indicesSize,
                               cudaMemcpyHostToDevice));

    const uint32_t triangleInputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.numVertices = static_cast<uint32_t>(meshVertices.size());
    triangleInput.triangleArray.vertexBuffers = &dVertices;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.numIndexTriplets = static_cast<uint32_t>(meshIndices.size() / 3);
    triangleInput.triangleArray.indexBuffer = dIndices;
    triangleInput.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
    triangleInput.triangleArray.flags = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions gasAccelOptions = {};
    gasAccelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    gasAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes;
    OTK_ERROR_CHECK(optixAccelComputeMemoryUsage(
        context, &gasAccelOptions, &triangleInput, 1, &gasBufferSizes));

    CUdeviceptr dTempBufferGas = 0;
    OTK_ERROR_CHECK(cudaMalloc(reinterpret_cast<void **>(&dTempBufferGas),
                               gasBufferSizes.tempSizeInBytes));

    OTK_ERROR_CHECK(cudaMalloc(reinterpret_cast<void **>(&dGasOutputBuffer),
                               gasBufferSizes.outputSizeInBytes));

    OTK_ERROR_CHECK_LOG(optixAccelBuild(
        context, 0, &gasAccelOptions, &triangleInput, 1,
        dTempBufferGas, gasBufferSizes.tempSizeInBytes,
        dGasOutputBuffer, gasBufferSizes.outputSizeInBytes,
        &gasHandle, nullptr, 0));

    // Free temporary buffers and the mesh input (the GAS now owns the geometry).
    OTK_ERROR_CHECK(cudaFree(reinterpret_cast<void *>(dTempBufferGas)));
    OTK_ERROR_CHECK(cudaFree(reinterpret_cast<void *>(dVertices)));
    OTK_ERROR_CHECK(cudaFree(reinterpret_cast<void *>(dIndices)));
    return 0;
}