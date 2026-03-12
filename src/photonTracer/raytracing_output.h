// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include "photontracer.h"

struct BufferDescriptor
{
    enum class ElementType : int
    {
        Float,
        Double,
        Float3,
        Float4,
        Int32,
        UInt32,
        String,
    };
    OutputType type;
    std::string name;
    size_t elementSize;
    ElementType elementType;
    mutable bool enabled;
    size_t bufferDimXOverride;
    bool zeroInitialize;

    BufferDescriptor(OutputType t, const std::string &n, size_t size, ElementType elemType, bool en = true, size_t bufferDimX = 0, bool zeroInit = false)
        : type(t), name(n), elementSize(size), elementType(elemType), enabled(en), bufferDimXOverride(bufferDimX), zeroInitialize(zeroInit) {}
};

class RayTracingOutput
{
public:
    RayTracingOutput();

    ~RayTracingOutput() = default;

    const BufferDescriptor &getDescriptor(OutputType type) const;

    const std::vector<BufferDescriptor> &getDescriptors() const;

    void setBufferDimX(OutputType type, size_t bufferDimX);

    /**
     * Configure output buffers.
     * @param type The type of the output buffer.
     * @param enabled Whether to enable or disable the buffer.
     */
    void enableOutput(OutputType type, bool enabled = true);

    /**
     * Check if a specific output buffer is enabled.
     * @param type The type of the output buffer.
     * @returns True if the buffer is enabled, false otherwise.
     */
    bool isOutputEnabled(OutputType type) const;

    uint32_t getOutputFlags() const;

private:
    std::vector<BufferDescriptor> bufferDescriptors_;
};