// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <cuda_runtime.h>
#include <optix.h>
#include <vector>
#include <string>
#include "materials.h"
#include "launch_types.h"
#pragma once

enum GeometryType
{
    MESH,
    MESH_INSTANCED,
};

enum LengthUnit
{
    MICRO_METER,
    MILLI_METER,
    METER
};


struct InputParameters
{
    unsigned int numberOfRays;
    float4 stokesVector;             // Stokes vector: [I, Q, U, V]
    float3 qMinusAxisSeed;           // Q- axis seed for the Stokes vector
    unsigned int maxScatteringCount; // Maximum scatteringCount for ray tracing; if 0, no limit
    unsigned int initSeed;
    bool useComplexFresnel;
    GeometryType geometryType;
    float lengthScale;
    OptixTraversableHandle handle;
    RayGeneratorType rayGeneratorType;
    RayGeneratorData rayGeneratorData;
    DeviceOutputBuffers deviceOutputBuffers;
    uint32_t outputFlags;
    uint32_t healpixNside;
    uint32_t healpixBinCount;
};

struct InputParametersSampleDensity
{
    unsigned int numberOfRays;
    unsigned int initSeed;
    float3 boxMin;
    float3 boxMax;
    OptixTraversableHandle handle;
    int32_t *intersectionCountBuffer;
};

struct RayGenData
{
    // No data needed
};

struct MissData
{
    // No data needed
};


struct HitGroupDataDensity
{
    // No data needed
};