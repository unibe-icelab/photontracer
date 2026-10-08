// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <limits>
#include <stdexcept>

#include "output_buffers.h"
#include "raytracing_output.h"

OutputBuffers::OutputBuffers(IRaytracingBackend &backend, const uint3 launchShape, const std::vector<BufferDescriptor> &descriptors)
    : backend_(&backend), launchShape_(launchShape), bufferDescriptors_(descriptors)
{
    for (const auto &descriptor : bufferDescriptors_)
    {
        if (!descriptor.enabled)
        {
            continue;
        }

        BufferLayout layout = {};
        layout.shape = launchShape_;

        if (descriptor.bufferDimXOverride != 0)
        {
            if (descriptor.bufferDimXOverride > static_cast<size_t>(std::numeric_limits<unsigned int>::max()))
            {
                throw std::runtime_error("Buffer element count override exceeds uint32 range");
            }
            layout.shape.x = static_cast<unsigned int>(descriptor.bufferDimXOverride);
        }

        const size_t elementCount = static_cast<size_t>(layout.shape.x) * layout.shape.y * layout.shape.z;
        const size_t bufferSize = elementCount * descriptor.elementSize;

        if (bufferSize > 0)
        {
            layout.devicePtr = backend_->allocateBuffer(bufferSize);
            if (descriptor.zeroInitialize)
            {
                backend_->clearBuffer(layout.devicePtr, bufferSize);
            }
        }
        else
        {
            layout.devicePtr = nullptr;
        }

        buffers_[descriptor.type] = layout;
    }
}

OutputBuffers::~OutputBuffers()
{
    for (auto &buffer : buffers_)
    {
        if (buffer.second.devicePtr)
        {
            backend_->freeBuffer(buffer.second.devicePtr);
        }
    }
    buffers_.clear();
}

const BufferLayout &OutputBuffers::getBufferLayout(OutputType type) const
{
    auto it = buffers_.find(type);
    if (it == buffers_.end())
    {
        throw std::runtime_error("Buffer layout not found for the specified output type");
    }
    return it->second;
}

DeviceOutputBuffers OutputBuffers::getDeviceOutputBuffers() const
{
    DeviceOutputBuffers outputBuffers = {};
    outputBuffers.lastDirection = getBuffer<float3>(OutputType::LAST_DIRECTION);
    outputBuffers.lastPosition = getBuffer<float3>(OutputType::LAST_POSITION);
    outputBuffers.ray_state = getBuffer<int>(OutputType::RAY_STATE);
    outputBuffers.lastMediumID = getBuffer<int>(OutputType::LAST_MEDIUM_ID);
    outputBuffers.scatteringCount = getBuffer<unsigned int>(OutputType::SCATTERING_COUNT);
    outputBuffers.numberOfWarnings = getBuffer<unsigned int>(OutputType::NUMBER_OF_WARNINGS);
    outputBuffers.stokesVector = getBuffer<float4>(OutputType::STOKES_VECTOR);
    outputBuffers.stokesVectorIn = getBuffer<float4>(OutputType::STOKES_VECTOR_IN);
    outputBuffers.opticalPathLength = getBuffer<double>(OutputType::OPTICAL_PATH_LENGTH);
    outputBuffers.sourceDirection = getBuffer<float3>(OutputType::SOURCE_DIRECTION);
    outputBuffers.sourcePosition = getBuffer<float3>(OutputType::SOURCE_POSITION);
    outputBuffers.scatteringAngle = getBuffer<float>(OutputType::SCATTERING_ANGLE);
    outputBuffers.qMinusAxisIn = getBuffer<float3>(OutputType::Q_MINUS_AXIS_IN);
    outputBuffers.directionHistogramHealpix = getBuffer<uint32_t>(OutputType::DIRECTION_HISTOGRAM_HEALPIX);

    return outputBuffers;
}

void OutputBuffers::clearZeroInitializedBuffers() const
{
    for (const auto &descriptor : bufferDescriptors_)
    {
        if (!descriptor.enabled || !descriptor.zeroInitialize)
        {
            continue;
        }

        auto it = buffers_.find(descriptor.type);
        if (it == buffers_.end())
        {
            throw std::runtime_error("Zero-initialized buffer '" + descriptor.name + "' was not allocated");
        }

        const BufferLayout &layout = it->second;
        const size_t elementCount = static_cast<size_t>(layout.shape.x) * layout.shape.y * layout.shape.z;
        if (elementCount == 0 || !layout.devicePtr)
        {
            continue;
        }

        const size_t bufferSize = elementCount * descriptor.elementSize;
        backend_->clearBuffer(layout.devicePtr, bufferSize);
    }
}

void OutputBuffers::copyToHost(OutputType type, void *host) const
{
    const BufferLayout &layout = getBufferLayout(type);
    for (const auto &descriptor : bufferDescriptors_)
    {
        if (descriptor.type == type)
        {
            const size_t bytes = static_cast<size_t>(layout.shape.x) * layout.shape.y * layout.shape.z * descriptor.elementSize;
            if (bytes > 0 && layout.devicePtr)
            {
                backend_->copyBufferToHost(host, layout.devicePtr, bytes);
            }
            return;
        }
    }
    throw std::runtime_error("Buffer descriptor not found for the specified output type");
}