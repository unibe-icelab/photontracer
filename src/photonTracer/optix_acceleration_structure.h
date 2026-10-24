// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include "i_geometry.h"

/// Device buffer and traversable handle of a geometry built by the OptiX backend
class OptixAccelerationStructure : public AccelerationStructure
{
public:
    using AccelerationStructure::AccelerationStructure;
    ~OptixAccelerationStructure() override
    {
        if (buffer)
        {
            cudaFree(reinterpret_cast<void *>(buffer));
        }
    }

    OptixTraversableHandle handle = 0;
    CUdeviceptr buffer = 0;
};
