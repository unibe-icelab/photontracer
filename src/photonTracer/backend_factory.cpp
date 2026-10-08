// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <stdexcept>

#include "backend_factory.h"

#ifdef PHOTONTRACER_HAS_OPTIX
#include <cuda_runtime.h>

#include "optix_raytracing_backend.h"
#endif

#ifdef PHOTONTRACER_HAS_EMBREE
#include "embree_raytracing_backend.h"
#endif

std::vector<BackendType> availableBackends()
{
    std::vector<BackendType> backends;
#ifdef PHOTONTRACER_HAS_OPTIX
    backends.push_back(BackendType::OPTIX);
#endif
#ifdef PHOTONTRACER_HAS_EMBREE
    backends.push_back(BackendType::EMBREE);
#endif
    return backends;
}

std::unique_ptr<IRaytracingBackend> makeBackend(BackendType type, const BackendOptions &options)
{
    switch (type)
    {
    case BackendType::OPTIX:
#ifdef PHOTONTRACER_HAS_OPTIX
        return std::make_unique<OptixRaytracingBackend>(options.gpuId, options.optixLoggingLevel, options.enableValidationMode);
#else
        throw std::runtime_error("This build of photontracer does not contain the OptiX backend");
#endif
    case BackendType::EMBREE:
#ifdef PHOTONTRACER_HAS_EMBREE
        return std::make_unique<EmbreeRaytracingBackend>(options.cpuThreads);
#else
        throw std::runtime_error("This build of photontracer does not contain the Embree backend");
#endif
    }
    throw std::invalid_argument("Unknown backend");
}

bool isCudaAvailable()
{
#ifdef PHOTONTRACER_HAS_OPTIX
    int deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
#else
    return false;
#endif
}
