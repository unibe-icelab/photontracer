// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include "photontracer.h"
#include "raytracing_output.h"

struct BufferLayout
{
    void *devicePtr;
    uint3 shape;
};

class OutputBuffers
{
public:
    OutputBuffers(uint3 bufferShape, const std::vector<BufferDescriptor> &descriptors);

    ~OutputBuffers();

    DeviceOutputBuffers getDeviceOutputBuffers() const;

    template <typename T>
    T *getBuffer(OutputType type) const;

    void *getPointer(OutputType type) { return getBuffer<void>(type); }
    const void *getPointer(OutputType type) const { return getBuffer<void>(type); }

    const uint3 &getConfiguredLaunchShape() const { return launchShape_; }
    const BufferLayout &getBufferLayout(OutputType type) const;

    void clearZeroInitializedBuffers() const;

private:
    uint3 launchShape_;
    std::unordered_map<OutputType, BufferLayout> buffers_;
    std::vector<BufferDescriptor> bufferDescriptors_;
};

template <typename T>
T *OutputBuffers::getBuffer(OutputType type) const
{
    auto it = buffers_.find(type);
    if (it != buffers_.end())
    {
        return static_cast<T *>(it->second.devicePtr);
    }
    return nullptr;
}