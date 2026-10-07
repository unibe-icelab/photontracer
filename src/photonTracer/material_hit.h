// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include "portable_math.h"
#include "complex_f.h"
#include "light_scattering.h"
#include "materials.h"
#include "ray_context.h"

// Kernel logging; a no-op unless logging.cuh was included first
#ifndef DBG_LOG_TEXT
#define DBG_LOG_STRING(label, text) \
    do                              \
    {                               \
    } while (0)
#define DBG_LOG_INT(label, value) \
    do                            \
    {                             \
    } while (0)
#define DBG_LOG_BOOL(label, value) \
    do                             \
    {                              \
    } while (0)
#define DBG_LOG_FLOAT(label, value) \
    do                              \
    {                               \
    } while (0)
#define DBG_LOG_FLOAT3(label, value) \
    do                               \
    {                                \
    } while (0)
#define DBG_LOG_TEXT(text) \
    do                     \
    {                      \
    } while (0)
#endif

// What a backend has to report about the surface a ray hit
struct HitInfo
{
    float3 rayOrigin;
    float3 rayDirection;
    float maxDistance;       // distance to the hit along rayDirection
    float3 hitPoint;         // position of the hit
    float3 front;            // spawn point just outside the surface
    float3 back;             // spawn point just inside the surface
    float3 worldNormal;      // surface normal, facing against the ray
    unsigned int instanceId; // material of the hit surface
    bool isFrontFace;
};

/**
 * @brief Moves a ray through the medium it travels in and the surface it hit:
 * absorption and volume scattering on the way, then diffuse, reflective or
 * Fresnel interaction at the surface. The result is written to the ray context.
 *
 * @param nextSample Returns a uniform random number in (0, 1].
 */
template <typename RayContextT, typename NextSampleT>
PT_INLINE PT_HD void handleMaterialHit(
    RayContextT &ctx, const HitInfo &hit, const HitGroupData &hgData,
    float lengthScale, bool useComplexFresnel, NextSampleT &nextSample)
{
    const float3 rayOrigin = hit.rayOrigin;
    float3 rayDir = hit.rayDirection;
    const float maxDistance = hit.maxDistance;
    const float3 front = hit.front;
    const float3 back = hit.back;
    const float3 worldNormal = hit.worldNormal;
    const unsigned int instanceId = hit.instanceId;
    const bool isFrontFace = hit.isFrontFace;

#if !defined(NDEBUG)
    if (instanceId >= MAX_MATERIALS)
    {
        printf("Error: instanceId %u out of bounds [0, 16)\n", instanceId);
        RayState state = ctx.getState();
        state.absorbed = 3;
        state.done = 1;
        ctx.setState(state);
        return;
    }
#endif

    const Material scatteringMaterial = hgData.materials[instanceId];
    DBG_LOG_INT("Hit material id", instanceId);
    DBG_LOG_INT("Hit material type", scatteringMaterial.type);

    float3 hitPoint = hit.hitPoint;

    DBG_LOG_FLOAT3("Hit point", hitPoint);
    DBG_LOG_FLOAT3("Normal", worldNormal);
    DBG_LOG_BOOL("Front face", isFrontFace);

    RayState state = ctx.getState();
#if !defined(NDEBUG)
    if (state.currentMedium >= MAX_MATERIALS)
    {
        printf("Error: current medium index %u out of bounds [0, 16)\n", state.currentMedium);
        state.absorbed = 3;
        state.done = 1;
        ctx.setState(state);
        return;
    }
#endif
    MaterialType currentMaterialType = hgData.materials[state.currentMedium].type;
    DBG_LOG_INT("Current medium", currentMaterialType);

    RefractiveIndex currentRefractiveIndex = {1.0f, 0.0f};
    bool didAbsorb = false;
    bool didScatter = false;
    float rndSample = nextSample();
    float travelledDistance = 0.0f;
    const float wavelengthUm = hgData.wavelengthUm;

    if (currentMaterialType == REFRACTIVE)
    {
        currentRefractiveIndex = hgData.materials[state.currentMedium].properties.refractive.refractiveIndex;
        DBG_LOG_TEXT("Current medium is refractive material");
        didAbsorb = calculateAbsorption(
            maxDistance,
            wavelengthUm,
            lengthScale,
            currentRefractiveIndex.i,
            rndSample,
            travelledDistance);
    }
    else if (currentMaterialType == VOLUME_SCATTERING)
    {
        currentRefractiveIndex = hgData.materials[state.currentMedium].properties.volumeScattering.refractiveIndex;
        DBG_LOG_TEXT("Current medium is volume scattering material");
        float absorptionCoefficient = absorptionCoefficientFromk(currentRefractiveIndex.i, wavelengthUm, lengthScale);
        DBG_LOG_FLOAT("Absorption coefficient", absorptionCoefficient);
        float scatteingCoefficient = hgData.materials[state.currentMedium].properties.volumeScattering.scatteringCoefficient;
        DBG_LOG_FLOAT("Scattering coefficient", scatteingCoefficient);
        float extinctionCoefficient = absorptionCoefficient + scatteingCoefficient;
        bool didAbsorbOrScatter = calculateDistance(
            maxDistance,
            extinctionCoefficient,
            rndSample,
            travelledDistance);

        if (didAbsorbOrScatter)
        {
            float singleScatteringAlbedo = scatteingCoefficient / extinctionCoefficient;
            DBG_LOG_FLOAT("Single scattering albedo", singleScatteringAlbedo);
            rndSample = nextSample();
            if (rndSample < singleScatteringAlbedo)
            {
                didScatter = true;
            }
            else
            {
                didAbsorb = true;
            }
            DBG_LOG_BOOL("Did scatter", didScatter);
            DBG_LOG_BOOL("Did absorb", didAbsorb);
        }
    }
    else
    {
        DBG_LOG_TEXT("Current medium is not absorbing/volume scattering");
        travelledDistance = maxDistance;
    }

    ctx.setLastSegmentOpticalPathLength(travelledDistance * currentRefractiveIndex.r);

    if (didAbsorb || didScatter)
    {
        // volume interaction
        hitPoint = rayOrigin + travelledDistance * rayDir;

        ctx.setOrigin(hitPoint);
        if (didAbsorb)
        {
            state.absorbed = 1;
            state.done = 1;
            ctx.setState(state);
            DBG_LOG_FLOAT("Absorbed distance", travelledDistance);
            DBG_LOG_INT("Absorbed medium", state.currentMedium);
        }
        else
        {
            DBG_LOG_FLOAT("Scattered distance", travelledDistance);
            DBG_LOG_INT("Scattered medium", state.currentMedium);

            float u1 = nextSample();
            float u2 = nextSample();
            auto newDirection = henyeyGreensteinDirection(
                rayDir,
                hgData.materials[state.currentMedium].properties.volumeScattering.asymetryParameter,
                u1,
                u2);

            ctx.setDirection(newDirection);
        }
    }
    else if (scatteringMaterial.type == DIFFUSE)
    {
        DBG_LOG_FLOAT("Diffuse albedo", scatteringMaterial.properties.diffuse.albedo);

        auto albedo = scatteringMaterial.properties.diffuse.albedo;

        if (nextSample() < albedo)
        {
            hitPoint = front;
            float u1 = nextSample();
            float u2 = nextSample();
            auto newDirection = calculateLamberianDirection(u1, u2, worldNormal);

            ctx.setOrigin(hitPoint);
            ctx.setDirection(newDirection);
            ctx.setState(state);
            ctx.setStokesVector(make_float4(1.0f, 0.0f, 0.0f, 0.0f));
        }
        else
        {
            ctx.setOrigin(hitPoint);
            state.absorbed = 1;
            state.done = 1;
            ctx.setState(state);
        }
    }
    else if (scatteringMaterial.type == REFLECTIVE)
    {
        DBG_LOG_FLOAT("Reflectivity", scatteringMaterial.properties.reflective.reflectivity);
        DBG_LOG_FLOAT("Fuzziness", scatteringMaterial.properties.reflective.fuzziness);
        auto reflectivity = scatteringMaterial.properties.reflective.reflectivity;
        auto fuzziness = scatteringMaterial.properties.reflective.fuzziness;

        if (nextSample() <= reflectivity)
        {
            hitPoint = front;

            auto newDirection = calculateReflectedDirection(rayDir, worldNormal);
            if (fuzziness > 0.0f)
            {
                float3 randomInUnitSphere;
                do
                {
                    randomInUnitSphere = make_float3(
                        2.0f * nextSample() - 1.0f,
                        2.0f * nextSample() - 1.0f,
                        2.0f * nextSample() - 1.0f);
                } while (otk::dot(randomInUnitSphere, randomInUnitSphere) >= 1.0f);
                float3 randomOnUnitSphere = otk::normalize(randomInUnitSphere);
                newDirection = otk::normalize(newDirection + fuzziness * randomOnUnitSphere);
                // Catch degenerate case where fuzziness is too high and newDirection is opposite to the normal
                if (otk::dot(newDirection, worldNormal) < 0.0f)
                {
                    newDirection = calculateReflectedDirection(rayDir, worldNormal);
                }
            }
            ctx.setOrigin(hitPoint);
            ctx.setDirection(newDirection);
            ctx.setState(state);
            ctx.setStokesVector(make_float4(1.0f, 0.0f, 0.0f, 0.0f));
        }
        else
        {
            ctx.setOrigin(hitPoint);
            state.absorbed = 1;
            state.done = 1;
            ctx.setState(state);
        }
    }
    else
    {
        DBG_LOG_TEXT("Hit refractive or volume scattering material");
        float4 stokesVector = ctx.getStokesVector();
        float3 qMinusAxis = ctx.getQMinusAxis();

        unsigned int nextMedium;

        if (isFrontFace)
        {
            nextMedium = instanceId; // Current instance is the next medium
        }
        else
        {
            if (state.currentMediumHistorySize == 0)
            {
                nextMedium = 0;
            }
            else
            {
                if (state.currentMedium == instanceId)
                { // If the current medium is the same as the instance, we need to step back
                    if (state.currentMediumHistorySize < 2)
                    {
                        nextMedium = 0; // If we are at the first layer, we go back to the default medium
                    }
                    else
                    {
                        nextMedium = ctx.getMedium(state.currentMediumHistorySize - 2); // Get the medium one layer back
                    }
                }
                else
                {
                    nextMedium = state.currentMedium; // If not, two different media are overlapping, so stay in the current medium
                    state.numberOfWarnings++;
                }
            }
        }
#if !defined(NDEBUG)
        if (nextMedium >= MAX_MATERIALS)
        {
            printf("Error: next medium index %u out of bounds [0, 16)\n", nextMedium);
            state.absorbed = 3;
            state.done = 1;
            ctx.setState(state);
            return;
        }
#endif
        MaterialType nextMaterialType = hgData.materials[nextMedium].type;
        RefractiveIndex nextRefractiveIndex = {1.0f, 0.0f};
        if (nextMaterialType == REFRACTIVE)
        {
            nextRefractiveIndex = hgData.materials[nextMedium].properties.refractive.refractiveIndex;
        }
        else if (nextMaterialType == VOLUME_SCATTERING)
        {
            nextRefractiveIndex = hgData.materials[nextMedium].properties.volumeScattering.refractiveIndex;
        }
        else
        {
            printf("Error: Next medium is not refractive or volume scattering material\n");
        }

        Complexf nA = Complexf(currentRefractiveIndex.r, currentRefractiveIndex.i);
        Complexf nB = Complexf(nextRefractiveIndex.r, nextRefractiveIndex.i);

        DBG_LOG_FLOAT("Current index real", currentRefractiveIndex.r);
        DBG_LOG_FLOAT("Current index imag", currentRefractiveIndex.i);
        DBG_LOG_FLOAT("Next index real", nextRefractiveIndex.r);
        DBG_LOG_FLOAT("Next index imag", nextRefractiveIndex.i);

        // material boundary interaction
        rndSample = nextSample();
        bool isReflected;
        if (useComplexFresnel)
        {
            isReflected = calculateFresnelInteraction(stokesVector, qMinusAxis, rayDir, worldNormal, nA, nB, rndSample);
        }
        else
        {
            isReflected = calculateFresnelInteraction(stokesVector, qMinusAxis, rayDir, worldNormal, nA.real(), nB.real(), rndSample);
        }
        DBG_LOG_TEXT(isReflected ? "Fresnel: Reflection" : "Fresnel: Transmission");
        if (!isReflected)
        {
            if (isFrontFace)
            {
                DBG_LOG_INT("Entering medium", instanceId);
                bool success = ctx.appendMedium(instanceId, state.currentMediumHistorySize);
                if (!success)
                {
                    state.numberOfWarnings++;
                    DBG_LOG_TEXT("Warning: medium history size exceeded limit");
                }
                state.currentMedium = instanceId;
            }
            else
            {
                DBG_LOG_INT("Exiting medium", instanceId);
                auto found = ctx.removeLastOccurence(instanceId, state.currentMediumHistorySize);
                if (!found)
                {
                    state.numberOfWarnings++;
                    DBG_LOG_TEXT("Warning: medium not found in history");
                }
                state.currentMedium = nextMedium;
            }
        }

        hitPoint = isReflected ? front : back;

        ctx.setOrigin(hitPoint);
        ctx.setDirection(rayDir);
        ctx.setStokesVector(stokesVector);
        ctx.setQMinusAxis(qMinusAxis);
        ctx.setState(state);
    }
}
