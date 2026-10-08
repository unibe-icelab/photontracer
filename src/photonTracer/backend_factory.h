// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <memory>
#include <vector>

#include "i_raytracing_backend.h"

enum class BackendType
{
    OPTIX,  ///< NVIDIA GPU
    EMBREE, ///< CPU
};

struct BackendOptions
{
    int gpuId = 0; ///< OptiX only
    int optixLoggingLevel = 1; ///< OptiX only
    bool enableValidationMode = false; ///< OptiX only
    uint32_t cpuThreads = 0; ///< Embree only; 0 uses all cores
};

/// The backends this build contains, the preferred one first
std::vector<BackendType> availableBackends();

/// Creates a backend; throws if this build does not contain it.
std::unique_ptr<IRaytracingBackend> makeBackend(BackendType type, const BackendOptions &options);

/// Whether a CUDA device is present; always false in builds without OptiX
bool isCudaAvailable();
