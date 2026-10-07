// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <iomanip>
#include <iostream>

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

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
    geometry.build(context_);
}

void OptixRaytracingBackend::initializePipeline(const std::vector<Material> &materials, float wavelengthUm, uint32_t maxTraversableGraphDepth)
{
    pipeline_ = std::make_unique<OptixRayTracingPipeline>(context_, materials, wavelengthUm, maxTraversableGraphDepth);
}

void OptixRaytracingBackend::updateShaderBindingTable(const std::vector<Material> &materials, float wavelengthUm)
{
    pipeline_->updateShaderBindingTable(materials, wavelengthUm);
}

void OptixRaytracingBackend::launch(InputParameters &params, const IGeometry &geometry, uint3 launchShape)
{
    params.handle = geometry.getTraversableHandle();
    pipeline_->launch(params, launchShape);
}

void OptixRaytracingBackend::initializeDensityPipeline(uint32_t maxTraversableGraphDepth)
{
    densityPipeline_ = std::make_unique<OptixVolumeFractionPipeline>(context_, maxTraversableGraphDepth);
}

void OptixRaytracingBackend::launchDensity(InputParametersSampleDensity &params, const IGeometry &geometry)
{
    params.handle = geometry.getTraversableHandle();
    densityPipeline_->launch(params);
}
