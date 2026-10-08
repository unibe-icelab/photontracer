// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <algorithm>

#include "trace_buffers.h"

TraceBuffers::TraceBuffers(IRaytracingBackend &backend, const std::vector<uint32_t> &rays, uint32_t maxSteps)
    : backend_(&backend), rays_(rays), maxSteps_(maxSteps)
{
    records_ = static_cast<TraceRecord *>(backend_->allocateBuffer(rays_.size() * maxSteps_ * sizeof(TraceRecord)));
    try
    {
        stepCounts_ = static_cast<uint32_t *>(backend_->allocateBuffer(rays_.size() * sizeof(uint32_t)));
    }
    catch (...)
    {
        backend_->freeBuffer(records_);
        throw;
    }
}

TraceBuffers::~TraceBuffers()
{
    backend_->freeBuffer(records_);
    backend_->freeBuffer(stepCounts_);
}

TraceParams TraceBuffers::prepare()
{
    backend_->clearBuffer(records_, rays_.size() * maxSteps_ * sizeof(TraceRecord));
    backend_->clearBuffer(stepCounts_, rays_.size() * sizeof(uint32_t));

    TraceParams params = {};
    params.rayCount = static_cast<uint32_t>(rays_.size());
    params.maxSteps = maxSteps_;
    std::copy(rays_.begin(), rays_.end(), params.rays);
    params.records = records_;
    params.stepCounts = stepCounts_;
    return params;
}

std::vector<TraceRecord> TraceBuffers::read() const
{
    std::vector<uint32_t> counts(rays_.size());
    backend_->copyBufferToHost(counts.data(), stepCounts_, counts.size() * sizeof(uint32_t));

    std::vector<TraceRecord> all(rays_.size() * maxSteps_);
    backend_->copyBufferToHost(all.data(), records_, all.size() * sizeof(TraceRecord));

    std::vector<TraceRecord> steps;
    for (size_t slot = 0; slot < rays_.size(); ++slot)
    {
        const size_t count = std::min<size_t>(counts[slot], maxSteps_);
        steps.insert(steps.end(), all.begin() + slot * maxSteps_, all.begin() + slot * maxSteps_ + count);
    }
    return steps;
}
