// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <vector>
#include <chrono>
#include <span>
#include <cassert>

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include "geometry_util.h"
#include "../i_geometry.h"
#include "instance_geometry.h"
#include "mesh_geometry.h"

InstanceGeometry::InstanceGeometry(
    std::vector<std::shared_ptr<IGeometry>> subGeometries,
    std::vector<float> instanceTransforms,
    std::vector<unsigned int> particleTypeIds,
    std::vector<unsigned int> materialIds)
    : subGeometries(std::move(subGeometries)),
      instanceTransforms(std::move(instanceTransforms)),
      particleTypeIds(std::move(particleTypeIds)),
      materialIds(std::move(materialIds)) {}

void InstanceGeometry::freeDeviceMemory()
{
    for (auto &subGeometry : subGeometries)
    {
        if (subGeometry)
        {
            subGeometry->freeDeviceMemory();
        }
    }

    IGeometry::freeDeviceMemory();
}

void InstanceGeometry::build(OptixDeviceContext &context)
{
    uint32_t maxInstancesPerIAS;
    optixDeviceContextGetProperty(
        context,
        OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,
        &maxInstancesPerIAS,
        sizeof(maxInstancesPerIAS));
    size_t numberOfInstances = instanceTransforms.size() / 12; // Each instance transform is 3x4 matrix (12 floats)
    std::vector<OptixInstance> allInstances(numberOfInstances);

    assert(numberOfInstances <= maxInstancesPerIAS &&
           ("Number of instances exceeds the maximum allowed per Instance Acceleration Structure on this device which is: " + std::to_string(maxInstancesPerIAS)).c_str());

    for (auto &subGeometry : subGeometries)
    {
        if (!subGeometry->isBuilt())
        {
            subGeometry->build(context);
        }
    }
    for (size_t i = 0; i < numberOfInstances; ++i)
    {
        OptixInstance instance = {};
        // Copy the 3x4 transform (12 floats per instance)
        for (int j = 0; j < 12; ++j)
        {
            instance.transform[j] = instanceTransforms[i * 12 + j];
        }

        instance.instanceId = materialIds[i];
        instance.sbtOffset = 0;
        instance.visibilityMask = 255;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        // Reference the AS built for this subgeometry.
        if (particleTypeIds[i] >= subGeometries.size())
        {
            throw std::runtime_error("Particle type ID out of bounds for subGeometries.");
        }
        instance.traversableHandle = subGeometries[particleTypeIds[i]]->getTraversableHandle();
        allInstances[i] = instance;
    }

    CUdeviceptr dInstanceBuffer = 0;
    size_t instanceBufferSize = numberOfInstances * sizeof(OptixInstance);
    OTK_ERROR_CHECK(cudaMalloc(reinterpret_cast<void **>(&dInstanceBuffer), instanceBufferSize));
    OTK_ERROR_CHECK(cudaMemcpy(reinterpret_cast<void *>(dInstanceBuffer),
                               allInstances.data(), instanceBufferSize,
                               cudaMemcpyHostToDevice));

    OptixBuildInput instanceInput = {};
    instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceInput.instanceArray.instances = dInstanceBuffer;
    instanceInput.instanceArray.numInstances = numberOfInstances;

    OptixAccelBuildOptions iasAccelOptions = {};
    iasAccelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    iasAccelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes iasBufferSizes;
    OTK_ERROR_CHECK(optixAccelComputeMemoryUsage(
        context, &iasAccelOptions, &instanceInput, 1, &iasBufferSizes));

    CUdeviceptr dTempBufferIas = 0;
    OTK_ERROR_CHECK(cudaMalloc(reinterpret_cast<void **>(&dTempBufferIas),
                               iasBufferSizes.tempSizeInBytes));

    OTK_ERROR_CHECK(cudaMalloc(reinterpret_cast<void **>(&dGeometryBuffer),
                               iasBufferSizes.outputSizeInBytes));

    OTK_ERROR_CHECK_LOG(optixAccelBuild(
        context, 0, &iasAccelOptions, &instanceInput, 1,
        dTempBufferIas, iasBufferSizes.tempSizeInBytes,
        dGeometryBuffer, iasBufferSizes.outputSizeInBytes,
        &traversableHandle, nullptr, 0));

    OTK_ERROR_CHECK(cudaFree(reinterpret_cast<void *>(dInstanceBuffer)));
    OTK_ERROR_CHECK(cudaFree(reinterpret_cast<void *>(dTempBufferIas)));
    // The top-level traversable handle now becomes the IAS handle.
    accelerationStructureBuilt = true;
}
GeometryType InstanceGeometry::getType() const
{
    return MESH_INSTANCED;
}