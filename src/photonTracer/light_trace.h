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
