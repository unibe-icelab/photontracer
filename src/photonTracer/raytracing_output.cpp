// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include "photontracer.h"
#include "raytracing_output.h"
#include "output_buffers.h"

RayTracingOutput::RayTracingOutput()
{
    // Initialize buffer descriptors (order should match OutputType enum)
    bufferDescriptors_ = {
        {OutputType::LAST_DIRECTION, "lastDirection", sizeof(float3), BufferDescriptor::ElementType::Float3, false},
        {OutputType::LAST_POSITION, "lastPosition", sizeof(float3), BufferDescriptor::ElementType::Float3, false},
        {OutputType::RAY_STATE, "ray_state", sizeof(int), BufferDescriptor::ElementType::Int32, false},
        {OutputType::LAST_MEDIUM_ID, "lastMediumID", sizeof(int), BufferDescriptor::ElementType::Int32, false},
        {OutputType::SCATTERING_COUNT, "scatteringCount", sizeof(unsigned int), BufferDescriptor::ElementType::UInt32, false},
        {OutputType::NUMBER_OF_WARNINGS, "numberOfWarnings", sizeof(unsigned int), BufferDescriptor::ElementType::UInt32, false},
        {OutputType::STOKES_VECTOR, "stokesVector", sizeof(float4), BufferDescriptor::ElementType::Float4, false},
        {OutputType::STOKES_VECTOR_IN, "stokesVectorIn", sizeof(float4), BufferDescriptor::ElementType::Float4, false},
        {OutputType::OPTICAL_PATH_LENGTH, "opticalPathLength", sizeof(double), BufferDescriptor::ElementType::Double, false},
        {OutputType::SOURCE_DIRECTION, "sourceDirection", sizeof(float3), BufferDescriptor::ElementType::Float3, false},
        {OutputType::SOURCE_POSITION, "sourcePosition", sizeof(float3), BufferDescriptor::ElementType::Float3, false},
        {OutputType::SCATTERING_ANGLE, "scatteringAngle", sizeof(float), BufferDescriptor::ElementType::Float, false},
        {OutputType::Q_MINUS_AXIS_IN, "qMinusAxisIn", sizeof(float3), BufferDescriptor::ElementType::Float3, false},
        {OutputType::LOGS, "logs", LOG_BYTES_PER_RAY * sizeof(char), BufferDescriptor::ElementType::String, false},
        {OutputType::LOG_OFFSETS, "logOffsets", sizeof(uint32_t), BufferDescriptor::ElementType::UInt32, false, 0, true},
        {OutputType::DIRECTION_HISTOGRAM_HEALPIX, "directionHistogramHealpix", sizeof(uint32_t), BufferDescriptor::ElementType::UInt32, false, 0, true},
    };

    const auto expectedSize = static_cast<size_t>(OutputType::OUTPUT_TYPE_COUNT);
    if (bufferDescriptors_.size() != expectedSize)
    {
        throw std::runtime_error("Buffer descriptor table and OutputType enum are out of sync");
    }
}

const BufferDescriptor &RayTracingOutput::getDescriptor(OutputType type) const
{
    const auto index = static_cast<size_t>(type);
    if (index >= bufferDescriptors_.size())
    {
        throw std::out_of_range("Invalid OutputType value");
    }
    return bufferDescriptors_[index];
}

const std::vector<BufferDescriptor> &RayTracingOutput::getDescriptors() const
{
    return bufferDescriptors_;
}

void RayTracingOutput::setBufferDimX(OutputType type, size_t bufferDimX)
{
    const auto index = static_cast<size_t>(type);
    if (index >= bufferDescriptors_.size())
    {
        throw std::out_of_range("Invalid OutputType value");
    }
    bufferDescriptors_[index].bufferDimXOverride = bufferDimX;
}

void RayTracingOutput::enableOutput(OutputType type, bool enabled)
{
    getDescriptor(type).enabled = enabled;
}

bool RayTracingOutput::isOutputEnabled(OutputType type) const
{
    return getDescriptor(type).enabled;
}

uint32_t RayTracingOutput::getOutputFlags() const
{
    uint32_t flags = 0;
    for (const auto &descriptor : bufferDescriptors_)
    {
        if (descriptor.enabled)
        {
            flags |= (1u << static_cast<uint32_t>(descriptor.type));
        }
    }
    return flags;
}