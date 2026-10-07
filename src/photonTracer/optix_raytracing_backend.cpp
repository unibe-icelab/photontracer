// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <iomanip>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include "optix_geometry_build.h"
#include "optix_raytracing_backend.h"

OptixRaytracingBackend::OptixRaytracingBackend(int gpuId, int optixLoggingLevel, bool enableValidationMode)
{
    int deviceCount = 0;
    OTK_ERROR_CHECK(cudaGetDeviceCount(&deviceCount));

    if (gpuId >= 0 && gpuId < deviceCount)
    {
        std::cout << "Selecting GPU device ID: " << gpuId + 1
                  << "/" << deviceCount << std::endl;
        OTK_ERROR_CHECK(cudaSetDevice(gpuId));

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gpuId);
        std::cout << "Using device: " << prop.name << std::endl;
    }
    else
    {
        std::cerr << "Warning: Invalid device ID " << gpuId
                  << ". Using default device." << std::endl;
    }

    // Initialize the CUDA context
    OTK_ERROR_CHECK(cudaFree(0));
    OTK_ERROR_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &contextLogCb;
    options.logCallbackLevel = optixLoggingLevel;
    if (enableValidationMode)
    {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    }

    CUcontext cuCtx = 0; // zero means take the current context
    OTK_ERROR_CHECK(optixDeviceContextCreate(cuCtx, &options, &context_));
}

OptixRaytracingBackend::~OptixRaytracingBackend()
{
    // The pipelines must go before the context they were created from
    pipeline_.reset();
    densityPipeline_.reset();
    if (context_)
    {
        optixDeviceContextDestroy(context_);
    }
}

void OptixRaytracingBackend::contextLogCb(uint32_t level, const char *tag, const char *message, void * /*cbdata*/)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

uint32_t OptixRaytracingBackend::getDeviceLimit(OptixDeviceProperty property) const
{
    uint32_t value = 0;
    optixDeviceContextGetProperty(context_, property, &value, sizeof(value));
    return value;
}

uint32_t OptixRaytracingBackend::getMaxTraversableGraphDepth() const
{
    return getDeviceLimit(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH);
}

uint32_t OptixRaytracingBackend::getMaxSubGeometries() const
{
    return getDeviceLimit(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS);
}

uint32_t OptixRaytracingBackend::getMaxMeshTriangles() const
{
    return getDeviceLimit(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS);
}

void *OptixRaytracingBackend::allocateBuffer(size_t bytes)
{
    void *buffer = nullptr;
    OTK_ERROR_CHECK(cudaMalloc(&buffer, bytes));
    return buffer;
}

void OptixRaytracingBackend::freeBuffer(void *buffer) noexcept
{
    cudaFree(buffer);
}

void OptixRaytracingBackend::clearBuffer(void *buffer, size_t bytes)
{
    OTK_ERROR_CHECK(cudaMemset(buffer, 0, bytes));
}

void OptixRaytracingBackend::copyBufferToHost(void *host, const void *buffer, size_t bytes)
{
    OTK_ERROR_CHECK(cudaMemcpy(host, buffer, bytes, cudaMemcpyDeviceToHost));
}

void OptixRaytracingBackend::buildGeometry(IGeometry &geometry)
{
    auto accelerationStructure = std::make_unique<OptixAccelerationStructure>(static_cast<const IRaytracingBackend *>(this));
    switch (geometry.getType())
    {
    case MESH:
        buildMesh(static_cast<const MeshGeometry &>(geometry), *accelerationStructure);
        break;
    case MESH_INSTANCED:
        buildInstances(static_cast<const InstanceGeometry &>(geometry), *accelerationStructure);
        break;
    default:
        throw std::runtime_error("Unknown geometry type");
    }
    geometry.setAccelerationStructure(std::move(accelerationStructure));
}

void OptixRaytracingBackend::buildMesh(const MeshGeometry &mesh, OptixAccelerationStructure &accelerationStructure)
{
    buildGasFromMesh(
        mesh.getVertices(), mesh.getIndices(), context_,
        accelerationStructure.handle, accelerationStructure.buffer, mesh.isCompact());
}

void OptixRaytracingBackend::buildInstances(const InstanceGeometry &instances, OptixAccelerationStructure &accelerationStructure)
{
    const uint32_t maxInstancesPerIAS = getMaxSubGeometries();
    const std::vector<float> &transforms = instances.getInstanceTransforms();
    const size_t numberOfInstances = transforms.size() / 12; // Each instance transform is 3x4 matrix (12 floats)

    assert(numberOfInstances <= maxInstancesPerIAS &&
           ("Number of instances exceeds the maximum allowed per Instance Acceleration Structure on this device which is: " + std::to_string(maxInstancesPerIAS)).c_str());

    const auto &subGeometries = instances.getSubGeometries();
    for (const auto &subGeometry : subGeometries)
    {
        if (!subGeometry->isBuiltBy(static_cast<const IRaytracingBackend *>(this)))
        {
            buildGeometry(*subGeometry);
        }
    }

    std::vector<OptixInstance> allInstances(numberOfInstances);
    for (size_t i = 0; i < numberOfInstances; ++i)
    {
        OptixInstance instance = {};
        // Copy the 3x4 transform (12 floats per instance)
        for (int j = 0; j < 12; ++j)
        {
            instance.transform[j] = transforms[i * 12 + j];
        }

        instance.instanceId = instances.getMaterialIds()[i];
        instance.sbtOffset = 0;
        instance.visibilityMask = 255;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        // Reference the AS built for this subgeometry.
        const unsigned int particleTypeId = instances.getParticleTypeIds()[i];
        if (particleTypeId >= subGeometries.size())
        {
            throw std::runtime_error("Particle type ID out of bounds for subGeometries.");
        }
        instance.traversableHandle = traversableHandle(*subGeometries[particleTypeId]);
        allInstances[i] = instance;
    }

    buildIasFromInstances(allInstances, context_, accelerationStructure.handle, accelerationStructure.buffer);
}

OptixTraversableHandle OptixRaytracingBackend::traversableHandle(const IGeometry &geometry) const
{
    if (!geometry.isBuiltBy(static_cast<const IRaytracingBackend *>(this)))
    {
        throw std::runtime_error("The geometry has not been built by this backend");
    }
    return static_cast<const OptixAccelerationStructure *>(geometry.getAccelerationStructure())->handle;
}

void OptixRaytracingBackend::initializePipeline(const std::vector<Material> &materials, float wavelengthUm, uint32_t maxTraversableGraphDepth,
                                                RandomNumberGenerator generator)
{
    pipeline_ = std::make_unique<OptixRayTracingPipeline>(context_, materials, wavelengthUm, maxTraversableGraphDepth, generator);
}

void OptixRaytracingBackend::updateShaderBindingTable(const std::vector<Material> &materials, float wavelengthUm)
{
    pipeline_->updateShaderBindingTable(materials, wavelengthUm);
}

void OptixRaytracingBackend::launch(InputParameters &params, const IGeometry &geometry, uint3 launchShape)
{
    params.handle = traversableHandle(geometry);
    pipeline_->launch(params, launchShape);
}

void OptixRaytracingBackend::initializeDensityPipeline(uint32_t maxTraversableGraphDepth, RandomNumberGenerator generator)
{
    densityPipeline_ = std::make_unique<OptixVolumeFractionPipeline>(context_, maxTraversableGraphDepth, generator);
}

void OptixRaytracingBackend::launchDensity(InputParametersSampleDensity &params, const IGeometry &geometry)
{
    params.handle = traversableHandle(geometry);
    densityPipeline_->launch(params);
}
