// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include "portable_math.h"
#include "light_scattering.h"

struct RayState
{
    uint32_t numberOfWarnings;         // 21 bit (0-2097151)
    uint32_t currentMedium;            // 4 bit (0-15)
    uint32_t currentMediumHistorySize; // 4 bit (0-15)
    uint32_t absorbed;                 // 2 bit (0-15)
    bool done;                         // 1 bit
};

struct RayData
{
    float3 direction;
    float3 origin;
    float4 stokesVector;
    float3 qMinusAxis;
    double opticalPathLength;
    RayState state;
    uint32_t packedMediumHistory; // 8 layers, 4 bits each
};

/**
 * The material interaction reads and writes the ray through a context object,
 * so the same code runs on an OptiX payload and on a plain struct in memory.
 * This is the in-memory version.
 */
struct MemoryRayContext
{
    RayData *ray;

    PT_HD float3 getOrigin() const { return ray->origin; }
    PT_HD void setOrigin(float3 origin) const { ray->origin = origin; }

    PT_HD float3 getDirection() const { return ray->direction; }
    PT_HD void setDirection(float3 direction) const { ray->direction = direction; }

    PT_HD float4 getStokesVector() const { return ray->stokesVector; }
    PT_HD void setStokesVector(float4 stokesVector) const { ray->stokesVector = stokesVector; }

    PT_HD float3 getQMinusAxis() const { return ray->qMinusAxis; }
    PT_HD void setQMinusAxis(float3 qMinusAxis) const { ray->qMinusAxis = qMinusAxis; }

    PT_HD RayState getState() const { return ray->state; }
    PT_HD void setState(RayState state) const { ray->state = state; }

    PT_HD uint32_t getMedium(uint32_t index) const { return getMediumFromPacked(ray->packedMediumHistory, index); }

    PT_HD bool appendMedium(uint32_t medium, uint32_t &historySize) const
    {
        return appendMediumPacked(medium, historySize, ray->packedMediumHistory);
    }

    PT_HD bool removeLastOccurence(uint32_t medium, uint32_t &historySize) const
    {
        return removeLastOccurencePacked(medium, historySize, ray->packedMediumHistory);
    }

    PT_HD void setLastSegmentOpticalPathLength(float length) const { ray->opticalPathLength += length; }
};
