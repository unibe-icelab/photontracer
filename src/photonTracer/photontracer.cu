// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

// Portions of this code were derived from NVIDIA OptiX sample code.
// See THIRD_PARTY_NOTICES.md for full license text.

#include <optix.h>
#include <optix_device.h>
#include <cstdint>

#include "photontracer.h"
#include "ray_payload.cuh"

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance.h>
#include "logging.cuh"
#include "light_scattering.h"
#include "material_hit.h"
#include "ray_generation.h"
#include "light_trace.h"
#include "complex_f.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C"
{
    __constant__ InputParameters params;
    __constant__ InputParametersSampleDensity paramsDensity;
}

static __forceinline__ __device__ void traceRay(
    OptixTraversableHandle handle,
    OptixRayData &rayData,
    float tmin,
    float tmax)
{
    unsigned int dirPayload[3];
    unsigned int origPayload[3];
    unsigned int stokPayload[4];
    unsigned int qDirPayLoad[3];
    unsigned int statePayload;
    unsigned int mediumHistoryPayload;
    unsigned int randStatePayload[6] = {}; // words beyond RAND_STATE_WORDS stay unused
    unsigned int oplLastSegmentPayload = __float_as_uint(0.0f);

    packFloat3(rayData.direction, dirPayload);
    packFloat3(rayData.origin, origPayload);
    packFloat4(rayData.stokesVector, stokPayload);
    packFloat3(rayData.qMinusAxis, qDirPayLoad);

    statePayload = packRayState(rayData.state);
    mediumHistoryPayload = rayData.packedMediumHistory;

    packRandState(rayData.randState, randStatePayload);

    OptixRayFlags rayFlags = OPTIX_RAY_FLAG_NONE;

    optixTrace(
        handle,
        rayData.origin,
        rayData.direction,
        tmin,                     // Min intersection distance
        tmax,                     // Max intersection distance
        0.0f,                     // rayTime -- used for motion blur
        OptixVisibilityMask(255), // Specify always visible
        rayFlags,
        0, // SBT offset   -- See SBT discussion
        1, // SBT stride   -- See SBT discussion
        0, // missSBTIndex -- See SBT discussion
        origPayload[0], origPayload[1], origPayload[2],
        dirPayload[0], dirPayload[1], dirPayload[2],
        stokPayload[0], stokPayload[1], stokPayload[2], stokPayload[3],
        qDirPayLoad[0], qDirPayLoad[1], qDirPayLoad[2],
        statePayload, mediumHistoryPayload,
        randStatePayload[0], randStatePayload[1], randStatePayload[2],
        randStatePayload[3], randStatePayload[4], randStatePayload[5],
        oplLastSegmentPayload);

    rayData.direction = unpackFloat3(dirPayload);
    rayData.origin = unpackFloat3(origPayload);
    rayData.stokesVector = unpackFloat4(stokPayload);
    rayData.qMinusAxis = unpackFloat3(qDirPayLoad);
    rayData.state = unpackRayState(statePayload);
    double oplLastSegment = static_cast<double>(__uint_as_float(oplLastSegmentPayload));
    rayData.opticalPathLength += oplLastSegment;
    rayData.packedMediumHistory = mediumHistoryPayload;
    rayData.randState = unpackRandState(randStatePayload);
}

static __forceinline__ __device__ void traceDensity(
    OptixTraversableHandle handle,
    DensityData &rayData,
    float tmin,
    float tmax)
{
    unsigned int dirPayload[3];
    unsigned int origPayload[3];
    unsigned int frontFacesPayload;
    unsigned int backFacesPayload;
    unsigned int donePayload;

    packFloat3(rayData.direction, dirPayload);
    packFloat3(rayData.origin, origPayload);

    frontFacesPayload = rayData.numberOfFrontFaces;
    backFacesPayload = rayData.numberOfBackFaces;
    donePayload = rayData.done;

    OptixRayFlags rayFlags = OPTIX_RAY_FLAG_NONE;

    optixTrace(
        handle,
        rayData.origin,
        rayData.direction,
        tmin,                     // Min intersection distance
        tmax,                     // Max intersection distance
        0.0f,                     // rayTime -- used for motion blur
        OptixVisibilityMask(255), // Specify always visible
        rayFlags,
        0, // SBT offset   -- See SBT discussion
        1, // SBT stride   -- See SBT discussion
        0, // missSBTIndex -- See SBT discussion
        origPayload[0], origPayload[1], origPayload[2],
        dirPayload[0], dirPayload[1], dirPayload[2],
        frontFacesPayload, backFacesPayload, donePayload);

    rayData.direction = unpackFloat3(dirPayload);
    rayData.origin = unpackFloat3(origPayload);
    rayData.numberOfFrontFaces = frontFacesPayload;
    rayData.numberOfBackFaces = backFacesPayload;
    rayData.done = donePayload;
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx3 = optixGetLaunchIndex();
    const uint3 launchDims = optixGetLaunchDimensions();

    uint32_t idx = ptLinearizeLaunchIndex(idx3, launchDims);

    RandState randState = randInit(params.initSeed, idx);
    auto nextSample = [&randState]() { return randNext(randState); };

    float3 rayOrigin, incidentRayDirection;
    if (!computeRay(params.rayGeneratorType, params.rayGeneratorData, nextSample, idx3, idx, rayOrigin, incidentRayDirection))
    {
        printf("Error: Unknown ray generator type\n");
        return;
    }

    OptixRayData prd;
    float4 stokesIn;
    float3 initialQMinusAxis;
    initializeLightTraceRay(
        params.outputFlags, params.deviceOutputBuffers, idx, rayOrigin, incidentRayDirection,
        params.stokesVector, params.qMinusAxisSeed, nextSample, prd, stokesIn, initialQMinusAxis);
    prd.randState = randState;

    uint32_t scatteringCount = 0;

    DBG_LOG_INT("Ray index", idx);

    for (;;)
    {
        DBG_LOG_TEXT("____Start new ray____");
        DBG_LOG_INT("Scattering count", scatteringCount);
        DBG_LOG_INT("Done", prd.state.done);
        DBG_LOG_FLOAT3("Origin", prd.origin);
        DBG_LOG_FLOAT3("Direction", prd.direction);
        DBG_LOG_INT("Medium history size", prd.state.currentMediumHistorySize);
        if (params.maxScatteringCount > 0 && scatteringCount >= params.maxScatteringCount)
        {
            DBG_LOG_INT("Max scatteringCount reached", params.maxScatteringCount);
            prd.state.absorbed = 2;
            break;
        }

        traceRay(params.handle, prd, 0.0f, 1e16f);

        if (prd.state.done)
        {
            break;
        }
        scatteringCount++;
    }
    DBG_LOG_INT("Done flag", prd.state.done);
    DBG_LOG_INT("Absorbed state", prd.state.absorbed);
    DBG_LOG_INT("Final scatteringCount", scatteringCount);

    writeMainTraceOutputs(
        params.outputFlags, params.deviceOutputBuffers, idx, prd,
        incidentRayDirection, stokesIn, initialQMinusAxis, scatteringCount);
}

extern "C" __global__ void __raygen__density()
{
    // Lookup our location within the launch grid
    const uint3 idx3 = optixGetLaunchIndex();
    const uint3 launchDims = optixGetLaunchDimensions();

    uint32_t idx = idx3.x;

    RandState randState = randInit(paramsDensity.initSeed, idx);

    auto nextSample = [&randState]() { return randNext(randState); };

    float3 rayOrigin = computeDensitySampleOrigin(paramsDensity.boxMin, paramsDensity.boxMax, nextSample);
    float3 incidentRayDirection;
    incidentRayDirection = make_float3(0.0f, 0.0f, 1.0f);

    // prepare the payload
    DensityData prd;
    // Trace the ray against our scene hierarchy
    prd.direction = incidentRayDirection;
    prd.origin = rayOrigin;
    prd.numberOfBackFaces = 0;
    prd.numberOfFrontFaces = 0;
    prd.done = false;

    for (;;)
    {
        traceDensity(paramsDensity.handle, prd, 0.0f, 1e16f);

        if (prd.done)
        {
            break;
        }
    }

    paramsDensity.intersectionCountBuffer[idx] = prd.numberOfBackFaces - prd.numberOfFrontFaces;
}

extern "C" __global__ void __miss__ms()
{
    if ((params.outputFlags & OUT_DIRECTION_HISTOGRAM_HEALPIX))
    {
        float3 dir = getRayDirection();
        dir = otk::normalize(dir);

        const int healpixBinIdx = healpixAng2PixRing(static_cast<int>(params.healpixNside), dir);
        const uint32_t pixelX = optixGetLaunchIndex().y;
        const uint32_t pixelY = optixGetLaunchIndex().z;
        const uint32_t pixelCountX = optixGetLaunchDimensions().y;
        const uint32_t pixelCountY = optixGetLaunchDimensions().z;
        const uint32_t pixelIndex = pixelY * pixelCountX + pixelX;
        uint32_t histogramIdx = pixelIndex * params.healpixBinCount + healpixBinIdx;
        if (histogramIdx < params.healpixBinCount * pixelCountX * pixelCountY)
        {
            atomicAdd(&params.deviceOutputBuffers.directionHistogramHealpix[histogramIdx], 1);
        }
    }
    setDone(1);
}

extern "C" __global__ void __miss__density()
{
    optixSetPayload_8(1);
}

extern "C" __global__ void __closesthit__ch()
{
    HitGroupData *hgData = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());

    HitInfo hit;
    hit.rayOrigin = optixGetWorldRayOrigin();
    hit.rayDirection = optixGetWorldRayDirection();
    hit.maxDistance = optixGetRayTmax();

    hit.instanceId = optixGetInstanceId();
    if (hit.instanceId == UINT32_MAX)
    {
        hit.instanceId = 1; // a plain mesh uses material 1 for its inside
    }

    if (!optixIsTriangleHit())
    {
        printf("Error: Unknown geometry type intersected\n");
        return;
    }
    DBG_LOG_TEXT("Hit triangle primitive");

    float3 hitPoint = hit.rayOrigin + hit.maxDistance * hit.rayDirection;

    // Object normal from the triangle vertices
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int gasSbtIdx = optixGetSbtGASIndex();
    const unsigned int primIdx = optixGetPrimitiveIndex();

    hit.isFrontFace = optixIsTriangleFrontFaceHit();
    float3 vertices[3] = {};
    optixGetTriangleVertexData(
        gas,
        primIdx,
        gasSbtIdx,
        0,
        vertices);
    float3 objectNormal = otk::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
    objectNormal = otk::normalize(objectNormal);

    float3 objectHitPoint = optixTransformPointFromWorldToObjectSpace(hitPoint);
    float offset;

    SelfIntersectionAvoidance::getSafeTriangleSpawnOffset(
        objectHitPoint,
        objectNormal,
        offset,
        vertices[0],
        vertices[1],
        vertices[2],
        optixGetTriangleBarycentrics());

    float worldOffset;

    SelfIntersectionAvoidance::transformSafeSpawnOffset(
        hitPoint,
        hit.worldNormal,
        worldOffset,
        objectHitPoint,
        objectNormal,
        offset);

    SelfIntersectionAvoidance::offsetSpawnPoint(
        hit.front,
        hit.back,
        hitPoint,
        hit.worldNormal,
        worldOffset);
    hit.hitPoint = hitPoint;

    if (otk::dot(hit.worldNormal, hit.rayDirection) > 0.0f)
    {
        hit.worldNormal = -hit.worldNormal;
        float3 temp = hit.front;
        hit.front = hit.back;
        hit.back = temp;
    }

    RandState randState = getRandState();
    auto nextSample = [&randState]() { return randNext(randState); };

    OptixPayloadRayContext ctx;
    handleMaterialHit(ctx, hit, *hgData, params.lengthScale, params.useComplexFresnel, nextSample);

    setRandState(randState);
}

extern "C" __global__ void __closesthit__density()
{
    HitGroupData *hgData = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
    // Get ray information and calculate the hit point
    float3 rayOrigin = optixGetWorldRayOrigin();
    float3 rayDir = optixGetWorldRayDirection();
    float maxDistance = optixGetRayTmax();

    float3 hitPoint = rayOrigin + maxDistance * rayDir;

    // Calculate the object normal from the cross product of the triangle vertices
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int gasSbtIdx = optixGetSbtGASIndex();
    const unsigned int primIdx = optixGetPrimitiveIndex();

    float3 front, back;
    float3 objectNormal, worldNormal;

    bool isFrontFace;

    if (optixIsTriangleHit())
    {
        isFrontFace = optixIsTriangleFrontFaceHit();
        if (isFrontFace)
        {
            uint32_t currentFrontHits = optixGetPayload_6();
            currentFrontHits++;
            optixSetPayload_6(currentFrontHits);
        }
        else
        {
            uint32_t currentBackHits = optixGetPayload_7();
            currentBackHits++;
            optixSetPayload_7(currentBackHits);
        }
        float3 vertices[3] = {};
        optixGetTriangleVertexData(
            gas,
            primIdx,
            gasSbtIdx,
            0,
            vertices);
        objectNormal = otk::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
        objectNormal = otk::normalize(objectNormal);

        float3 objectHitPoint = optixTransformPointFromWorldToObjectSpace(hitPoint);
        float offset;

        SelfIntersectionAvoidance::getSafeTriangleSpawnOffset(
            objectHitPoint,
            objectNormal,
            offset,
            vertices[0],
            vertices[1],
            vertices[2],
            optixGetTriangleBarycentrics());

        float worldOffset;

        SelfIntersectionAvoidance::transformSafeSpawnOffset(
            hitPoint,
            worldNormal,
            worldOffset,
            objectHitPoint,
            objectNormal,
            offset);

        SelfIntersectionAvoidance::offsetSpawnPoint(
            front,
            back,
            hitPoint,
            worldNormal,
            worldOffset);

        // always continue straight, but avoid self intersection by offsetting the hit point
        if (otk::dot(worldNormal, rayDir) > 0.0f)
        {
            hitPoint = front;
        }
        else
        {
            hitPoint = back;
        }
    }
    else
    {
        printf("Error: Unknown geometry type intersected\n");
        return;
    }
    setRayOrigin(hitPoint);
}