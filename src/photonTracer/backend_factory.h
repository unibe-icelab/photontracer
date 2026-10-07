// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <memory>

#include "i_raytracing_backend.h"

/// Creates the raytracing backend of this build; throws if the build has none.
std::unique_ptr<IRaytracingBackend> makeBackend(int gpuId, int optixLoggingLevel, bool enableValidationMode);
