// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include "portable_math.h"
#include "launch_types.h"

// Ray sources, source polarization and the HEALPix binning, independent of the ray tracing backend.
// The random number source is a callable returning a uniform number in (0, 1]; the order in which
// the numbers are drawn is part of the behavior.

PT_INLINE PT_HD int floorToInt(float x)
{
#if defined(__CUDA_ARCH__)
    return __float2int_rd(x); // round toward -inf, matches std::floor for floats
#else
    return static_cast<int>(floorf(x));
#endif
}

/// HEALPix pixel (ring scheme) of a direction
PT_INLINE PT_HD int healpixAng2PixRing(int nside, const float3 d)
{
    const float twoPi = 2.0f * M_PI;
    const float invHalfPi = 2.0f / M_PI;

    const float z = d.z;
    const float za = fabsf(z);

    float phi = atan2f(d.y, d.x);
    if (phi < 0.0f)
        phi += twoPi;

    // tt spans [0,4)
    const float tt = phi * invHalfPi;

    const int nl2 = 2 * nside;
    const int nl4 = 4 * nside;
    const int ncap = nl2 * (nside - 1);

    int ipix = 0;

    if (za <= 2.0f / 3.0f)
    {
        const float temp1 = nside * (0.5f + tt);
        const float temp2 = nside * (z * 0.75f);

        const int jp = floorToInt(temp1 - temp2);
        const int jm = floorToInt(temp1 + temp2);

        const int ir = nside + 1 + jp - jm;
        const int kshift = 1 - (ir & 1);

        int ip = (jp + jm - nside + kshift + 1) >> 1;
        ip = (ip % nl4 + nl4) % nl4; // safe wrap

        ipix = ncap + (ir - 1) * nl4 + ip;
    }
    else
    {
        const float tp = tt - floorf(tt); // fractional part in [0,1)
        const float tmp = nside * sqrtf(3.0f * (1.0f - za));

        int jp = floorToInt(tp * tmp) + 1;
        int jm = floorToInt((1.0f - tp) * tmp) + 1;
        const int ir = jp + jm - 1;
        const int ip = floorToInt(tt * ir);

        if (z > 0.0f)
            ipix = 2 * ir * (ir - 1) + ip;
        else
            ipix = 12 * nside * nside - 2 * ir * (ir + 1) + ip;
    }

    return ipix;
}

/// Parallel beam: a disk of rays around the source origin
template <typename NextSampleT>
PT_INLINE PT_HD void computeRayParallel(const RayGeneratorData::ParallelSource &source, NextSampleT &nextSample, float3 &origin, float3 &direction)
{
    origin = source.origin;
    direction = source.direction;
    float sourceRadius = source.offsetRadius;

    // Compute orthogonal vectors u and v
    float3 u = make_float3(0.0f, 1.0f, 0.0f);
    if (fabsf(direction.y) > 0.999f)
    {
        u = make_float3(1.0f, 0.0f, 0.0f);
    }
    u = otk::normalize(otk::cross(direction, u));
    float3 v = otk::normalize(otk::cross(direction, u));

    float x, y;
    do
    {
        x = 2.0f * nextSample() - 1.0f;
        y = 2.0f * nextSample() - 1.0f;
    } while (x * x + y * y > 1.0f);

    origin += sourceRadius * (x * u + y * v);
}

/// Isotropic source: a point on a sphere around the center, aimed inward, with a disk offset
template <typename NextSampleT>
PT_INLINE PT_HD void computeRayIsotropic(const RayGeneratorData::IsotropicSource &source, NextSampleT &nextSample, float3 &origin, float3 &direction)
{
    origin = source.center;
    float sourceRadius = source.sourceRadius;
    float offsetRadius = source.offsetRadius;

    // Sample a point on the sphere of radius sourceRadius
    float x, y, z;
    do
    {
        x = 2.0f * nextSample() - 1.0f;
        y = 2.0f * nextSample() - 1.0f;
        z = 2.0f * nextSample() - 1.0f;
    } while (x * x + y * y + z * z > 1.0f);

    direction = otk::normalize(make_float3(x, y, z)); // Uniformly distributed on the unit sphere, outward pointing

    origin += sourceRadius * direction;

    // Compute orthogonal vectors u and v
    float3 u = make_float3(0.0f, 1.0f, 0.0f);
    if (fabsf(direction.y) > 0.999f)
    {
        u = make_float3(1.0f, 0.0f, 0.0f);
    }
    u = otk::normalize(otk::cross(direction, u));
    float3 v = otk::normalize(otk::cross(direction, u));

    do
    {
        x = 2.0f * nextSample() - 1.0f;
        y = 2.0f * nextSample() - 1.0f;
    } while (x * x + y * y > 1.0f);

    origin += offsetRadius * (x * u + y * v);
    direction = -direction; // Pointing inward
}

/// Diffuse source: a disk of rays around the origin, with cosine-weighted directions around the axis
template <typename NextSampleT>
PT_INLINE PT_HD void computeRayDiffuse(const RayGeneratorData::DiffuseSource &source, NextSampleT &nextSample, float3 &origin, float3 &direction)
{
    origin = source.origin;
    const float3 axis = source.direction;

    // Compute orthogonal vectors u and v
    float3 u = make_float3(0.0f, 1.0f, 0.0f);
    if (fabsf(axis.y) > 0.999f)
    {
        u = make_float3(1.0f, 0.0f, 0.0f);
    }
    u = otk::normalize(otk::cross(axis, u));
    float3 v = otk::normalize(otk::cross(axis, u));

    // Uniform point of the disk
    const float radius = source.offsetRadius * sqrtf(nextSample());
    const float angle = 2.0f * static_cast<float>(M_PI) * nextSample();
    origin += radius * (cosf(angle) * u + sinf(angle) * v);

    const float u1 = nextSample();
    const float u2 = nextSample();
    direction = cosineWeightedDirection(u1, u2, axis);
}

/// Linear ray index of a camera launch, whose launch index is (sample, pixel x, pixel y)
PT_INLINE PT_HD uint32_t cameraRayIndex(const RayGeneratorData::CameraSource &camera, uint3 launchIndex)
{
    const unsigned int samplesPerPixel = camera.samplesPerPixel > 0 ? camera.samplesPerPixel : 1;
    const unsigned int imageWidth = camera.imageWidth > 0 ? camera.imageWidth : 1;
    const unsigned int imageHeight = camera.imageHeight > 0 ? camera.imageHeight : 1;

    unsigned int sampleIndex = launchIndex.x;
    unsigned int pixelX = launchIndex.y;
    unsigned int pixelY = launchIndex.z;

    if (sampleIndex >= samplesPerPixel)
    {
        sampleIndex = samplesPerPixel - 1;
    }
    if (pixelX >= imageWidth)
    {
        pixelX = imageWidth - 1;
    }
    if (pixelY >= imageHeight)
    {
        pixelY = imageHeight - 1;
    }

    const uint32_t pixelIndex = pixelY * imageWidth + pixelX;
    return pixelIndex * samplesPerPixel + sampleIndex;
}

/// Pinhole camera with an optional defocus disk; rayIndex comes from cameraRayIndex()
template <typename NextSampleT>
PT_INLINE PT_HD void computeRayCamera(const RayGeneratorData::CameraSource &camera, NextSampleT &nextSample, uint32_t rayIndex, float3 &origin, float3 &direction)
{
    unsigned int samplesPerPixel = camera.samplesPerPixel > 0 ? camera.samplesPerPixel : 1;
    unsigned int imageWidth = camera.imageWidth > 0 ? camera.imageWidth : 1;
    unsigned int imageHeight = camera.imageHeight > 0 ? camera.imageHeight : 1;
    unsigned int pixelIndex = rayIndex / samplesPerPixel;
    const unsigned int pixelCount = imageWidth * imageHeight;
    if (pixelCount == 0)
    {
        pixelIndex = 0;
    }
    else if (pixelIndex >= pixelCount)
    {
        pixelIndex = pixelCount - 1;
    }
    unsigned int pixelX = pixelIndex % imageWidth;
    unsigned int pixelY = pixelIndex / imageWidth;

    float offsetX = nextSample() - 0.5f;
    float offsetY = nextSample() - 0.5f;

    float3 pixelSample = camera.pixel00;
    pixelSample += (static_cast<float>(pixelX) + offsetX) * camera.pixelDeltaU;
    pixelSample += (static_cast<float>(pixelY) + offsetY) * camera.pixelDeltaV;

    origin = camera.center;
    if (camera.enableDefocus)
    {
        float angle = 2.0f * M_PI * nextSample();
        float radius = sqrtf(nextSample());
        float x = cosf(angle) * radius;
        float y = sinf(angle) * radius;
        float3 defocusOffset = x * camera.defocusDiskU + y * camera.defocusDiskV;
        origin += defocusOffset;
    }

    direction = otk::normalize(pixelSample - origin);
}

/**
 * @brief Generates the ray of one launch index with the configured source.
 *
 * @param[in,out] rayIndex Linear launch index; replaced for a camera by the pixel and sample based index.
 * @return false if the generator type is unknown.
 */
template <typename NextSampleT>
PT_INLINE PT_HD bool computeRay(
    RayGeneratorType type, const RayGeneratorData &data, NextSampleT &nextSample,
    uint3 launchIndex, uint32_t &rayIndex, float3 &origin, float3 &direction)
{
    if (type == RAYGEN_PARALLEL)
    {
        computeRayParallel(data.parallel, nextSample, origin, direction);
    }
    else if (type == RAYGEN_ISOTROPIC)
    {
        computeRayIsotropic(data.isotropic, nextSample, origin, direction);
    }
    else if (type == RAYGEN_DIFFUSE)
    {
        computeRayDiffuse(data.diffuse, nextSample, origin, direction);
    }
    else if (type == RAYGEN_CAMERA)
    {
        rayIndex = cameraRayIndex(data.camera, launchIndex);
        computeRayCamera(data.camera, nextSample, rayIndex, origin, direction);
    }
    else
    {
        return false;
    }
    return true;
}

/// Source Stokes vector; NaN entries are replaced by a random linear (Q, U) or circular (V) polarization
template <typename NextSampleT>
PT_INLINE PT_HD float4 randomizeStokesVector(float4 stokesSetting, NextSampleT &nextSample)
{
    float4 stokesIn = stokesSetting;
    if (isnan(stokesSetting.y) || isnan(stokesSetting.z))
    {
        // random full linear polarization
        // random azimuthal angle psi in [0, pi]
        float rnd = nextSample();
        float psi = M_PI * rnd;
        stokesIn.y = cosf(2.0f * psi);
        stokesIn.z = sinf(2.0f * psi);
    }
    else if (isnan(stokesSetting.w))
    {
        // random circular polarization
        float rnd = nextSample();
        stokesIn.w = cosf(2.0f * M_PI * rnd);
    }
    return stokesIn;
}

/// Initial Q- axis: the seed projected perpendicular to the ray, or a fallback if there is no usable seed
PT_INLINE PT_HD float3 initialQMinusAxis(float3 qMinusAxisSeed, float3 k)
{
    float3 axis;
    if (!isnan(qMinusAxisSeed.x))
    {
        axis = otk::normalize(qMinusAxisSeed - otk::dot(qMinusAxisSeed, k) * k); // project and normalize
    }
    if (isnan(qMinusAxisSeed.x) || !isfinite(axis.x))
    { // not set or k ≈ seed: choose e_x or e_y as seed
        float3 up = make_float3(1.0f, 0.0f, 0.0f);
        axis = otk::normalize(up - otk::dot(up, k) * k); // project and normalize
        if (!isfinite(axis.x))
        { // k ≈ up: choose orthonormal e_y as seed
            up = make_float3(0.0f, 1.0f, 0.0f);
            axis = otk::normalize(up - otk::dot(up, k) * k);
        }
    }
    return axis;
}

/// Random point inside the box, for sampling the volume fraction
template <typename NextSampleT>
PT_INLINE PT_HD float3 computeDensitySampleOrigin(float3 boxMin, float3 boxMax, NextSampleT &nextSample)
{
    float r1 = nextSample();
    float r2 = nextSample();
    float r3 = nextSample();

    float3 boxSize = boxMax - boxMin;

    float3 origin;
    origin.x = r1 * boxSize.x + boxMin.x;
    origin.y = r2 * boxSize.y + boxMin.y;
    origin.z = r3 * boxSize.z + boxMin.z;
    return origin;
}
