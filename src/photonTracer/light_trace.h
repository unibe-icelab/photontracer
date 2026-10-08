// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include "portable_math.h"
#include "launch_types.h"
#include "ray_context.h"
#include "ray_generation.h"
#include "light_scattering.h"

// The part of a light trace that is the same for every backend: preparing the ray and
// writing the output buffers. Only the loop between the two differs per backend.

/**
 * @brief Prepares `prd` from a generated ray and writes the source outputs.
 *
 * @param[out] stokesIn Source Stokes vector, needed again by writeMainTraceOutputs().
 * @param[out] qMinusAxisIn Initial Q- axis, needed again by writeMainTraceOutputs().
 */
template <typename NextSampleT>
PT_INLINE PT_HD void initializeLightTraceRay(
    uint32_t outputFlags, const DeviceOutputBuffers &out, uint32_t idx,
    float3 rayOrigin, float3 rayDirection, float4 stokesSetting, float3 qMinusAxisSeed,
    NextSampleT &nextSample, RayData &prd, float4 &stokesIn, float3 &qMinusAxisIn)
{
    if (outputFlags & OUT_SOURCE_POSITION)
    {
        out.sourcePosition[idx] = rayOrigin;
    }
    if (outputFlags & OUT_SOURCE_DIRECTION)
    {
        out.sourceDirection[idx] = rayDirection;
    }

    prd.direction = rayDirection;
    prd.origin = rayOrigin;

    stokesIn = randomizeStokesVector(stokesSetting, nextSample);
    prd.stokesVector = stokesIn;

    qMinusAxisIn = initialQMinusAxis(qMinusAxisSeed, rayDirection);
    if (outputFlags & OUT_Q_MINUS_AXIS_IN)
    {
        out.qMinusAxisIn[idx] = qMinusAxisIn;
    }
    prd.qMinusAxis = qMinusAxisIn;

    prd.opticalPathLength = 0.0;
    prd.packedMediumHistory = 0;

    prd.state.numberOfWarnings = 0;
    prd.state.absorbed = 0;
    prd.state.done = 0;
    prd.state.currentMedium = 0;
    prd.state.currentMediumHistorySize = 0;
}

/// Writes the outputs of a finished ray
PT_INLINE PT_HD void writeMainTraceOutputs(
    uint32_t outputFlags, const DeviceOutputBuffers &out, uint32_t idx, const RayData &prd,
    float3 sourceDirection, float4 stokesIn, float3 qMinusAxisIn, uint32_t scatteringCount)
{
    if (outputFlags & OUT_LAST_DIRECTION)
    {
        out.lastDirection[idx] = prd.direction;
    }
    if (outputFlags & OUT_LAST_POSITION)
    {
        out.lastPosition[idx] = prd.origin;
    }
    if (outputFlags & OUT_RAY_STATE)
    {
        out.ray_state[idx] = prd.state.absorbed;
    }
    if (outputFlags & OUT_LAST_MEDIUM_ID)
    {
        out.lastMediumID[idx] = prd.state.currentMedium;
    }
    if (outputFlags & OUT_SCATTERING_COUNT)
    {
        out.scatteringCount[idx] = scatteringCount;
    }
    if (outputFlags & OUT_NUMBER_OF_WARNINGS)
    {
        out.numberOfWarnings[idx] = prd.state.numberOfWarnings;
    }
    if (outputFlags & OUT_STOKES_VECTOR)
    {
        float3 qMinusAxisScPlane;
        float4 rotatedStokesVector;
        scatteringPlaneNormalAxis(sourceDirection, prd.direction, qMinusAxisScPlane);
        if (!isnan(qMinusAxisScPlane.x))
        {
            float omega = signedRotationAboutAxis(prd.direction, prd.qMinusAxis, qMinusAxisScPlane);
            otk::Transform4 stokesRotationMatrix = calculateRotationMatrix(omega);
            rotatedStokesVector = stokesRotationMatrix * prd.stokesVector;
        }
        else
        {
            rotatedStokesVector = prd.stokesVector;
        }

        out.stokesVector[idx] = rotatedStokesVector;
    }
    if (outputFlags & OUT_STOKES_VECTOR_IN)
    {
        float4 initialStokes = stokesIn;
        float3 qMinusAxisScPlane;
        float4 rotatedStokesVector;

        scatteringPlaneNormalAxis(sourceDirection, prd.direction, qMinusAxisScPlane);
        if (!isnan(qMinusAxisScPlane.x))
        {
            float omega = signedRotationAboutAxis(sourceDirection, qMinusAxisIn, qMinusAxisScPlane);
            otk::Transform4 stokesRotationMatrix = calculateRotationMatrix(omega);
            rotatedStokesVector = stokesRotationMatrix * initialStokes;
        }
        else
        {
            rotatedStokesVector = initialStokes;
        }

        out.stokesVectorIn[idx] = rotatedStokesVector;
    }
    if (outputFlags & OUT_OPTICAL_PATH_LENGTH)
    {
        out.opticalPathLength[idx] = prd.opticalPathLength;
    }
    if (outputFlags & OUT_SCATTERING_ANGLE)
    {
        float cosTheta = otk::dot(sourceDirection, prd.direction);
        cosTheta = fmaxf(fminf(cosTheta, 1.0f), -1.0f); // Clamp to [-1, 1]
        out.scatteringAngle[idx] = acosf(cosTheta);
    }
}

/// Slot of a ray in the trace buffers, or -1 if it is not traced
PT_INLINE PT_HD int findTraceSlot(const TraceParams &trace, uint32_t ray)
{
    int low = 0;
    int high = static_cast<int>(trace.rayCount) - 1;
    while (low <= high)
    {
        const int middle = (low + high) / 2;
        if (trace.rays[middle] == ray)
        {
            return middle;
        }
        if (trace.rays[middle] < ray)
        {
            low = middle + 1;
        }
        else
        {
            high = middle - 1;
        }
    }
    return -1;
}

/// What a step did to the ray, from its state afterwards
PT_INLINE PT_HD uint32_t traceEventOf(const RayState &state)
{
    if (state.absorbed == 1)
    {
        return TRACE_ABSORBED;
    }
    if (state.absorbed == 3)
    {
        return TRACE_ERROR;
    }
    return state.done ? TRACE_ESCAPED : TRACE_INTERACTION;
}

/// Writes the start of a step of a traced ray, before the ray is traced; steps beyond `maxSteps` are
/// only counted.
PT_INLINE PT_HD void recordTraceStart(const TraceParams &trace, int slot, uint32_t ray, uint32_t step, const RayData &before)
{
    trace.stepCounts[slot] = step + 1;
    if (step >= trace.maxSteps)
    {
        return;
    }
    TraceRecord &record = trace.records[static_cast<size_t>(slot) * trace.maxSteps + step];
    record.ray = ray;
    record.step = step;
    record.mediumIn = before.state.currentMedium;
    record.originIn[0] = before.origin.x;
    record.originIn[1] = before.origin.y;
    record.originIn[2] = before.origin.z;
    record.directionIn[0] = before.direction.x;
    record.directionIn[1] = before.direction.y;
    record.directionIn[2] = before.direction.z;
}

/// Completes the step started by recordTraceStart() with the ray after it
PT_INLINE PT_HD void recordTraceEnd(const TraceParams &trace, int slot, uint32_t step, const RayData &after, uint32_t event)
{
    if (step >= trace.maxSteps)
    {
        return;
    }
    TraceRecord &record = trace.records[static_cast<size_t>(slot) * trace.maxSteps + step];
    record.event = event;
    record.mediumOut = after.state.currentMedium;
    record.originOut[0] = after.origin.x;
    record.originOut[1] = after.origin.y;
    record.originOut[2] = after.origin.z;
    record.directionOut[0] = after.direction.x;
    record.directionOut[1] = after.direction.y;
    record.directionOut[2] = after.direction.z;
    record.stokes[0] = after.stokesVector.x;
    record.stokes[1] = after.stokesVector.y;
    record.stokes[2] = after.stokesVector.z;
    record.stokes[3] = after.stokesVector.w;
    record.opticalPathLength = static_cast<float>(after.opticalPathLength);
}
