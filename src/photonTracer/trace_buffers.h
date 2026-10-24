// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <vector>

#include "i_raytracing_backend.h"
#include "launch_types.h"

/// Memory of the backend that the steps of the traced rays are written to
class TraceBuffers
{
public:
    /// @param rays Sorted indices of the traced rays.
    TraceBuffers(IRaytracingBackend &backend, const std::vector<uint32_t> &rays, uint32_t maxSteps);
    ~TraceBuffers();

    TraceBuffers(const TraceBuffers &) = delete;
    TraceBuffers &operator=(const TraceBuffers &) = delete;

    /// Sets the pointers and indices of the launch parameters and clears the buffers
    TraceParams prepare();

    /// The recorded steps, ordered by ray and step
    std::vector<TraceRecord> read() const;

private:
    IRaytracingBackend *backend_;
    std::vector<uint32_t> rays_;
    uint32_t maxSteps_;
    TraceRecord *records_ = nullptr;
    uint32_t *stepCounts_ = nullptr;
};
