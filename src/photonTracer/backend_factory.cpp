// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <stdexcept>

#include "backend_factory.h"

#ifdef PHOTONTRACER_HAS_OPTIX
#include "optix_raytracing_backend.h"
#endif

std::unique_ptr<IRaytracingBackend> makeBackend(int gpuId, int optixLoggingLevel, bool enableValidationMode)
{
#ifdef PHOTONTRACER_HAS_OPTIX
    return std::make_unique<OptixRaytracingBackend>(gpuId, optixLoggingLevel, enableValidationMode);
#else
    (void)gpuId;
    (void)optixLoggingLevel;
    (void)enableValidationMode;
    throw std::runtime_error("This build of photontracer has no raytracing backend");
#endif
}
