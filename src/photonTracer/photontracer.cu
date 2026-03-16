// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

// Portions of this code were derived from NVIDIA OptiX sample code.
// See THIRD_PARTY_NOTICES.md for full license text.

#include <optix.h>
#include <optix_device.h>
#include <curand_kernel.h>
#include <cstdint>

#include "photontracer.h"
#include "ray_payload.cuh"

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance.h>
#include "logging.cuh"
#include "light_scattering.cuh"
#include "complex_f.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C"
{
    __constant__ InputParameters params;
    __constant__ InputParametersSampleDensity paramsDensity;
}

static constexpr float TWOPI = 2.0f * M_PI;
static constexpr float INV_HALF_PI = 2.0f / M_PI; // 1 / (0.5 * PI)

__device__ __forceinline__ int floor_to_int(float x)
{
    return __float2int_rd(x); // round toward -inf, matches std::floor for floats
}

__device__ __forceinline__ int healpix_ang2pix_ring(int nside, const float3 d)
{
    const float z = d.z;
    const float za = fabsf(z);

    float phi = atan2f(d.y, d.x);
    if (phi < 0.0f)
        phi += TWOPI;

    // tt spans [0,4)
    const float tt = phi * INV_HALF_PI;

    const int nl2 = 2 * nside;
    const int nl4 = 4 * nside;
    const int ncap = nl2 * (nside - 1);

    int ipix = 0;

    if (za <= 2.0f / 3.0f)
    {
        const float temp1 = nside * (0.5f + tt);
        const float temp2 = nside * (z * 0.75f);

        const int jp = floor_to_int(temp1 - temp2);
        const int jm = floor_to_int(temp1 + temp2);

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

        int jp = floor_to_int(tp * tmp) + 1;
        int jm = floor_to_int((1.0f - tp) * tmp) + 1;
        const int ir = jp + jm - 1;
        const int ip = floor_to_int(tt * ir);

        if (z > 0.0f)
            ipix = 2 * ir * (ir - 1) + ip;
        else
            ipix = 12 * nside * nside - 2 * ir * (ir + 1) + ip;
    }

    return ipix;
}

static __forceinline__ __device__ void computeRayParallel(float3 &origin, float3 &direction, curandStateMRG32k3a *curandState)
{
    auto parallel_ray_gen_data = params.rayGeneratorData.parallel;
    origin = parallel_ray_gen_data.origin;
    direction = parallel_ray_gen_data.direction;
    float sourceRadius = parallel_ray_gen_data.offsetRadius;

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
        x = 2.0f * curand_uniform(curandState) - 1.0f;
        y = 2.0f * curand_uniform(curandState) - 1.0f;
    } while (x * x + y * y > 1.0f);

    origin += sourceRadius * (x * u + y * v);
}

// alternative method for isotropic ray generation: closer to SIRIS4
/*
static __forceinline__ __device__ void computeRayIsotropic(float3 &origin, float3 &direction, curandStateMRG32k3a *curandState)
{
    auto isotropic_ray_gen_data = params.rayGeneratorData.isotropic;
    origin = isotropic_ray_gen_data.center;
    float offsetRadius = isotropic_ray_gen_data.offsetRadius;

    // sample isotropic incoming direction d (mui uniform in [-1,1], phi uniform)
    float r = curand_uniform(curandState);
    float mui = 1.0f - 2.0f * r; // cos(theta)
    float phii = 2.0f * M_PI * curand_uniform(curandState);
    float nui = sqrtf(fmaxf(0.0f, 1.0f - mui * mui));
    float3 d;
    d.x = nui * cosf(phii);
    d.y = nui * sinf(phii);
    d.z = mui;
    d = otk::normalize(d); // unit vector pointing toward particle center

    // sample impact parameter uniformly in disk: r0 = offsetRadius * sqrt(U), phi0 uniform
    float r0 = offsetRadius * sqrtf(curand_uniform(curandState));
    float phi0 = 2.0f * M_PI * curand_uniform(curandState);

    // canonical start point (in canonical frame where canonical direction = +z)
    float3 X0;
    float x0 = r0 * cosf(phi0);
    float y0 = r0 * sinf(phi0);
    float z0 = -sqrtf(fmaxf(0.0f, offsetRadius * offsetRadius - r0 * r0));
    X0.x = x0;
    X0.y = y0;
    X0.z = z0;

    // rotate X0 from canonical +z frame so that +z -> d (Rodrigues rotation)
    // handle near-parallel and antiparallel cases
    float3 zaxis = make_float3(0.0f, 0.0f, 1.0f);
    float cosTheta = d.z; // dot(zaxis, d)
    float absCos = fabsf(cosTheta);

    float3 Xrot;
    if (absCos > 0.999999f)
    {
        // almost parallel: identity or 180 deg
        if (cosTheta > 0.0f)
        {
            Xrot = X0; // +z -> +z
        }
        else
        {
            // 180 degree rotation around X
            Xrot.x = X0.x;
            Xrot.y = -X0.y;
            Xrot.z = -X0.z;
        }
    }
    else
    {
        // general Rodrigues rotation: rotate by angle theta around axis k = z x d / |z x d|
        float3 k = make_float3(
            zaxis.y * d.z - zaxis.z * d.y,
            zaxis.z * d.x - zaxis.x * d.z,
            zaxis.x * d.y - zaxis.y * d.x);
        float s = sqrtf(k.x * k.x + k.y * k.y + k.z * k.z);
        k.x /= s;
        k.y /= s;
        k.z /= s;
        float sinTheta = s; // |z x d| = sin(theta)
        // Rodrigues: v_rot = v*cosθ + (k x v)*sinθ + k*(k·v)*(1-cosθ)
        float3 kxv = make_float3(
            k.y * X0.z - k.z * X0.y,
            k.z * X0.x - k.x * X0.z,
            k.x * X0.y - k.y * X0.x);
        float kdotv = k.x * X0.x + k.y * X0.y + k.z * X0.z;
        Xrot.x = X0.x * cosTheta + kxv.x * sinTheta + k.x * kdotv * (1.0f - cosTheta);
        Xrot.y = X0.y * cosTheta + kxv.y * sinTheta + k.y * kdotv * (1.0f - cosTheta);
        Xrot.z = X0.z * cosTheta + kxv.z * sinTheta + k.z * kdotv * (1.0f - cosTheta);
    }

    // set origin and direction in particle coordinates
    origin = isotropic_ray_gen_data.center + Xrot;
    direction = d; // points inward toward particle center (same as Fortran KEOUT)
}
ß*/

static __forceinline__ __device__ void computeRayIsotropicMC(float3 &origin, float3 &direction, curandStateMRG32k3a *curandState)
{
    auto isotropic_ray_gen_data = params.rayGeneratorData.isotropic;
    origin = isotropic_ray_gen_data.center;
    float sourceRadius = isotropic_ray_gen_data.sourceRadius;
    float offsetRadius = isotropic_ray_gen_data.offsetRadius;

    // Sample a point on the sphere of radius sourceRadius
    float x, y, z;
    do
    {
        x = 2.0f * curand_uniform(curandState) - 1.0f;
        y = 2.0f * curand_uniform(curandState) - 1.0f;
        z = 2.0f * curand_uniform(curandState) - 1.0f;
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
        x = 2.0f * curand_uniform(curandState) - 1.0f;
        y = 2.0f * curand_uniform(curandState) - 1.0f;
    } while (x * x + y * y > 1.0f);

    origin += offsetRadius * (x * u + y * v);
    direction = -direction; // Pointing inward
}

static __forceinline__ __device__ void computeRayCamera(float3 &origin, float3 &direction, curandStateMRG32k3a *curandState, uint32_t launchIndex)
{
    auto camera = params.rayGeneratorData.camera;
    unsigned int samplesPerPixel = camera.samplesPerPixel > 0 ? camera.samplesPerPixel : 1;
    unsigned int imageWidth = camera.imageWidth > 0 ? camera.imageWidth : 1;
    unsigned int imageHeight = camera.imageHeight > 0 ? camera.imageHeight : 1;
    unsigned int pixelIndex = launchIndex / samplesPerPixel;
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

    float offsetX = curand_uniform(curandState) - 0.5f;
    float offsetY = curand_uniform(curandState) - 0.5f;

    float3 pixelSample = camera.pixel00;
    pixelSample += (static_cast<float>(pixelX) + offsetX) * camera.pixelDeltaU;
    pixelSample += (static_cast<float>(pixelY) + offsetY) * camera.pixelDeltaV;

    origin = camera.center;
    if (camera.enableDefocus)
    {
        float angle = 2.0f * M_PI * curand_uniform(curandState);
        float radius = sqrtf(curand_uniform(curandState));
        float x = cosf(angle) * radius;
        float y = sinf(angle) * radius;
        float3 defocusOffset = x * camera.defocusDiskU + y * camera.defocusDiskV;
        origin += defocusOffset;
    }

    direction = otk::normalize(pixelSample - origin);
}

static __forceinline__ __device__ void traceRay(
    OptixTraversableHandle handle,
    RayData &rayData,
    float tmin,
    float tmax)
{
    unsigned int dirPayload[3];
    unsigned int origPayload[3];
    unsigned int stokPayload[4];
    unsigned int qDirPayLoad[3];
    unsigned int statePayload;
    unsigned int mediumHistoryPayload;
    unsigned int curandStatePayload[6];
    unsigned int oplLastSegmentPayload = __float_as_uint(0.0f);

    packFloat3(rayData.direction, dirPayload);
    packFloat3(rayData.origin, origPayload);
    packFloat4(rayData.stokesVector, stokPayload);
    packFloat3(rayData.qMinusAxis, qDirPayLoad);

    statePayload = packRayState(rayData.state);
    mediumHistoryPayload = packMediumHistory(rayData.mediumHistory);

    packCuRandStateMRG32k3a(&rayData.randState, curandStatePayload);

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
        curandStatePayload[0], curandStatePayload[1], curandStatePayload[2],
        curandStatePayload[3], curandStatePayload[4], curandStatePayload[5],
        oplLastSegmentPayload);

    rayData.direction = unpackFloat3(dirPayload);
    rayData.origin = unpackFloat3(origPayload);
    rayData.stokesVector = unpackFloat4(stokPayload);
    rayData.qMinusAxis = unpackFloat3(qDirPayLoad);
    rayData.state = unpackRayState(statePayload);
    double oplLastSegment = static_cast<double>(__uint_as_float(oplLastSegmentPayload));
    rayData.opticalPathLength += oplLastSegment;
    unpackMediumHistory(mediumHistoryPayload, rayData.mediumHistory);
    rayData.randState = unpackCuRandStateMRG32k3a(curandStatePayload);
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

    curandStateMRG32k3a curandState;
    curand_init(params.initSeed, idx, 0, &curandState);

    // Map our launch idx to a screen location and create a ray from the camera
    float3 rayOrigin, incidentRayDirection;
    if (params.rayGeneratorType == RAYGEN_PARALLEL)
    {
        computeRayParallel(rayOrigin, incidentRayDirection, &curandState);
    }
    else if (params.rayGeneratorType == RAYGEN_ISOTROPIC)
    {
        computeRayIsotropicMC(rayOrigin, incidentRayDirection, &curandState);
    }
    else if (params.rayGeneratorType == RAYGEN_CAMERA)
    {
        auto camera = params.rayGeneratorData.camera;
        const unsigned int samplesPerPixel = camera.samplesPerPixel > 0 ? camera.samplesPerPixel : 1;
        const unsigned int imageWidth = camera.imageWidth > 0 ? camera.imageWidth : 1;
        const unsigned int imageHeight = camera.imageHeight > 0 ? camera.imageHeight : 1;

        unsigned int sampleIndex = idx3.x;
        unsigned int pixelX = idx3.y;
        unsigned int pixelY = idx3.z;

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
        idx = pixelIndex * samplesPerPixel + sampleIndex;

        computeRayCamera(rayOrigin, incidentRayDirection, &curandState, idx);
    }
    else
    {
        printf("Error: Unknown ray generator type\n");
        return;
    }

    if (params.outputFlags & OUT_SOURCE_POSITION)
    {
        params.deviceOutputBuffers.sourcePosition[idx] = rayOrigin;
    }
    if (params.outputFlags & OUT_SOURCE_DIRECTION)
    {
        params.deviceOutputBuffers.sourceDirection[idx] = incidentRayDirection;
    }

    // prepare the payload
    RayData prd;
    // Trace the ray against our scene hierarchy
    prd.direction = incidentRayDirection;
    prd.origin = rayOrigin;

    auto stokesSetting = params.stokesVector;
    float4 stokesIn = stokesSetting;
    if (isnan(stokesSetting.y) || isnan(stokesSetting.z))
    {
        // random full linear polarization
        // random azimuthal angle psi in [0, pi]
        float rnd = curand_uniform(&curandState);
        float psi = M_PI * rnd;
        stokesIn.y = cosf(2.0f * psi);
        stokesIn.z = sinf(2.0f * psi);
    }
    else if (isnan(stokesSetting.w))
    {
        // random circular polarization
        float rnd = curand_uniform(&curandState);
        stokesIn.w = cosf(2.0f * M_PI * rnd);
    }

    prd.stokesVector = stokesIn;
    auto k = incidentRayDirection;

    float3 initialQMinusAxis;
    if (!isnan(params.qMinusAxisSeed.x))
    {
        initialQMinusAxis = otk::normalize(params.qMinusAxisSeed - otk::dot(params.qMinusAxisSeed, k) * k); // project and normalize
    }
    if (isnan(params.qMinusAxisSeed.x) || !isfinite(initialQMinusAxis.x))
    { // not set or k ≈ qMinusAxis: choose e_x or e_y as seed
        float3 up = make_float3(1.0f, 0.0f, 0.0f);
        initialQMinusAxis = otk::normalize(up - otk::dot(up, k) * k); // project and normalize
        if (!isfinite(initialQMinusAxis.x))
        { // k ≈ up: choose orthonormal e_y as seed
            up = make_float3(0.0f, 1.0f, 0.0f);
            initialQMinusAxis = otk::normalize(up - otk::dot(up, k) * k);
        }
    }

    if (params.outputFlags & OUT_Q_MINUS_AXIS_IN)
    {
        params.deviceOutputBuffers.qMinusAxisIn[idx] = initialQMinusAxis;
    }

    prd.qMinusAxis = initialQMinusAxis;
    prd.randState = curandState;
    prd.opticalPathLength = 0.0;

    prd.state.numberOfWarnings = 0;
    prd.state.absorbed = 0;
    prd.state.done = 0;

    prd.state.currentMedium = 0;
    prd.state.currentMediumHistorySize = 0;

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

    // check if buffer is allocated
    if (params.outputFlags & OUT_LAST_DIRECTION)
    {
        params.deviceOutputBuffers.lastDirection[idx] = prd.direction;
    }
    if (params.outputFlags & OUT_LAST_POSITION)
    {
        params.deviceOutputBuffers.lastPosition[idx] = prd.origin;
    }
    if (params.outputFlags & OUT_RAY_STATE)
    {
        params.deviceOutputBuffers.ray_state[idx] = prd.state.absorbed;
    }
    if (params.outputFlags & OUT_LAST_MEDIUM_ID)
    {
        params.deviceOutputBuffers.lastMediumID[idx] = prd.state.currentMedium;
    }
    if (params.outputFlags & OUT_SCATTERING_COUNT)
    {
        params.deviceOutputBuffers.scatteringCount[idx] = scatteringCount;
    }
    if (params.outputFlags & OUT_NUMBER_OF_WARNINGS)
    {
        params.deviceOutputBuffers.numberOfWarnings[idx] = prd.state.numberOfWarnings;
    }
    if (params.outputFlags & OUT_STOKES_VECTOR)
    {
        float3 qMinusAxisScPlane;
        float4 rotatedStokesVector;
        scatteringPlaneNormalAxis(incidentRayDirection, prd.direction, qMinusAxisScPlane);
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

        params.deviceOutputBuffers.stokesVector[idx] = rotatedStokesVector;
    }
    if (params.outputFlags & OUT_STOKES_VECTOR_IN)
    {
        float4 initialStokes = stokesIn;
        float3 qMinusAxisScPlane;
        float4 rotatedStokesVector;

        scatteringPlaneNormalAxis(incidentRayDirection, prd.direction, qMinusAxisScPlane);
        if (!isnan(qMinusAxisScPlane.x))
        {
            float omega = signedRotationAboutAxis(incidentRayDirection, initialQMinusAxis, qMinusAxisScPlane);
            otk::Transform4 stokesRotationMatrix = calculateRotationMatrix(omega);
            rotatedStokesVector = stokesRotationMatrix * initialStokes;
        }
        else
        {
            rotatedStokesVector = initialStokes;
        }

        params.deviceOutputBuffers.stokesVectorIn[idx] = rotatedStokesVector;
    }
    if (params.outputFlags & OUT_OPTICAL_PATH_LENGTH)
    {
        params.deviceOutputBuffers.opticalPathLength[idx] = prd.opticalPathLength;
    }
    if (params.outputFlags & OUT_SCATTERING_ANGLE)
    {
        float cosTheta = otk::dot(incidentRayDirection, prd.direction);
        cosTheta = fmaxf(fminf(cosTheta, 1.0f), -1.0f); // Clamp to [-1, 1]
        float scatteringAngle = acosf(cosTheta);
        params.deviceOutputBuffers.scatteringAngle[idx] = scatteringAngle;
    }
}

extern "C" __global__ void __raygen__density()
{
    // Lookup our location within the launch grid
    const uint3 idx3 = optixGetLaunchIndex();
    const uint3 launchDims = optixGetLaunchDimensions();

    uint32_t idx = idx3.x;

    curandStateMRG32k3a curandState;
    curand_init(paramsDensity.initSeed, idx, 0, &curandState);

    // Map our launch idx to a screen location and create a ray from the camera
    float3 rayOrigin, incidentRayDirection;
    float r1 = curand_uniform(&curandState);
    float r2 = curand_uniform(&curandState);
    float r3 = curand_uniform(&curandState);

    float3 boxSize = paramsDensity.boxMax - paramsDensity.boxMin;

    rayOrigin.x = r1 * boxSize.x + paramsDensity.boxMin.x;
    rayOrigin.y = r2 * boxSize.y + paramsDensity.boxMin.y;
    rayOrigin.z = r3 * boxSize.z + paramsDensity.boxMin.z;

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

        const int healpixBinIdx = healpix_ang2pix_ring(static_cast<int>(params.healpixNside), dir);
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
    // Get ray information and calculate the hit point
    float3 rayOrigin = optixGetWorldRayOrigin();
    float3 rayDir = optixGetWorldRayDirection();
    float maxDistance = optixGetRayTmax();

    unsigned int instanceId = optixGetInstanceId();

    if (instanceId == UINT32_MAX)
    {
        instanceId = 1;
    }

#if !defined(NDEBUG)
    if (instanceId < 0 || instanceId >= 15)
    {
        printf("Error: instanceId %u out of bounds [0, 15]\n", instanceId);
        RayState state = getRayState();
        state.absorbed = 3;
        state.done = 1;
        setRayState(state);
        return;
    }
#endif

    Material scatteringMaterial = hgData->materials[instanceId];
    DBG_LOG_INT("Hit material id", instanceId);
    DBG_LOG_INT("Hit material type", scatteringMaterial.type);

    float3 hitPoint = rayOrigin + maxDistance * rayDir;

    curandStateMRG32k3a curandState = getCuRandStateMRG32k3a();

    // Calculate the object normal from the cross product of the triangle vertices
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int gasSbtIdx = optixGetSbtGASIndex();
    const unsigned int primIdx = optixGetPrimitiveIndex();

    float3 front, back;
    float3 objectNormal, worldNormal;

    bool isFrontFace;

    if (optixIsTriangleHit())
    {
        DBG_LOG_TEXT("Hit triangle primitive");
        isFrontFace = optixIsTriangleFrontFaceHit();
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

        if (otk::dot(worldNormal, rayDir) > 0.0f)
        {
            worldNormal = -worldNormal;
            // swap front and back
            float3 temp = front;
            front = back;
            back = temp;
        }
    }
    else
    {
        printf("Error: Unknown geometry type intersected\n");
        return;
    }

    DBG_LOG_FLOAT3("Hit point", hitPoint);
    DBG_LOG_FLOAT3("Normal", worldNormal);
    DBG_LOG_BOOL("Front face", isFrontFace);

    RayState state = getRayState();
#if !defined(NDEBUG)
    if (state.currentMedium < 0 || state.currentMedium >= 15)
    {
        printf("Error: current medium index %d out of bounds [0, 15]\n", state.currentMedium);
        state.absorbed = 3;
        state.done = 1;
        setRayState(state);
        return;
    }
#endif
    MaterialType currentMaterialType = hgData->materials[state.currentMedium].type;
    DBG_LOG_INT("Current medium", currentMaterialType);

    RefractiveIndex currentRefractiveIndex = {1.0f, 0.0f};
    bool didAbsorb = false;
    bool didScatter = false;
    float rndSample = curand_uniform(&curandState);
    float travelledDistance = 0.0f;
    float wavelengthUm = hgData->wavelengthUm;

    if (currentMaterialType == REFRACTIVE)
    {
        currentRefractiveIndex = hgData->materials[state.currentMedium].properties.refractive.refractiveIndex;
        DBG_LOG_TEXT("Current medium is refractive material");
        didAbsorb = calculateAbsorption(
            maxDistance,
            wavelengthUm,
            params.lengthScale,
            currentRefractiveIndex.i,
            rndSample,
            travelledDistance);
    }
    else if (currentMaterialType == VOLUME_SCATTERING)
    {
        currentRefractiveIndex = hgData->materials[state.currentMedium].properties.volumeScattering.refractiveIndex;
        DBG_LOG_TEXT("Current medium is volume scattering material");
        float absorptionCoefficient = absorptionCoefficientFromk(currentRefractiveIndex.i, wavelengthUm, params.lengthScale);
        DBG_LOG_FLOAT("Absorption coefficient", absorptionCoefficient);
        float scatteingCoefficient = hgData->materials[state.currentMedium].properties.volumeScattering.scatteringCoefficient;
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
            rndSample = curand_uniform(&curandState);
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

    setOpticalPathLength(travelledDistance * currentRefractiveIndex.r);

    if (didAbsorb || didScatter)
    {
        // volume interaction
        hitPoint = rayOrigin + travelledDistance * rayDir;

        setRayOrigin(hitPoint);
        if (didAbsorb)
        {
            state.absorbed = 1;
            state.done = 1;
            setRayState(state);
            DBG_LOG_FLOAT("Absorbed distance", travelledDistance);
            DBG_LOG_INT("Absorbed medium", state.currentMedium);
        }
        else
        {
            DBG_LOG_FLOAT("Scattered distance", travelledDistance);
            DBG_LOG_INT("Scattered medium", state.currentMedium);

            float u1 = curand_uniform(&curandState);
            float u2 = curand_uniform(&curandState);
            auto newDirection = henyeyGreensteinDirection(
                rayDir,
                hgData->materials[state.currentMedium].properties.volumeScattering.asymetryParameter,
                u1,
                u2);

            setRayDirection(newDirection);
            setCuRandStateMRG32k3a(curandState);
        }
    }
    else if (scatteringMaterial.type == DIFFUSE)
    {
        DBG_LOG_FLOAT("Diffuse albedo", scatteringMaterial.properties.diffuse.albedo);

        auto albedo = hgData->materials[instanceId].properties.diffuse.albedo;

        if (curand_uniform(&curandState) < albedo)
        {
            hitPoint = front;
            float u1 = curand_uniform(&curandState);
            float u2 = curand_uniform(&curandState);
            auto newDirection = calculateLamberianDirection(u1, u2, worldNormal);

            setRayOrigin(hitPoint);
            setRayDirection(newDirection);
            setCuRandStateMRG32k3a(curandState);
            setRayState(state);
            setStokesVector(make_float4(1.0f, 0.0f, 0.0f, 0.0f));
        }
        else
        {
            setRayOrigin(hitPoint);
            state.absorbed = 1;
            state.done = 1;
            setRayState(state);
        }
    }
    else if (scatteringMaterial.type == REFLECTIVE)
    {
        DBG_LOG_FLOAT("Reflectivity", scatteringMaterial.properties.reflective.reflectivity);
        DBG_LOG_FLOAT("Fuzziness", scatteringMaterial.properties.reflective.fuzziness);
        auto reflectivity = hgData->materials[instanceId].properties.reflective.reflectivity;
        auto fuzziness = hgData->materials[instanceId].properties.reflective.fuzziness;

        if (curand_uniform(&curandState) <= reflectivity)
        {
            hitPoint = front;

            auto newDirection = calculateReflectedDirection(rayDir, worldNormal);
            if (fuzziness > 0.0f)
            {
                float3 randomInUnitSphere;
                do
                {
                    randomInUnitSphere = make_float3(
                        2.0f * curand_uniform(&curandState) - 1.0f,
                        2.0f * curand_uniform(&curandState) - 1.0f,
                        2.0f * curand_uniform(&curandState) - 1.0f);
                } while (otk::dot(randomInUnitSphere, randomInUnitSphere) >= 1.0f);
                float3 randomOnUnitSphere = otk::normalize(randomInUnitSphere);
                newDirection = otk::normalize(newDirection + fuzziness * randomOnUnitSphere);
                // Catch degenerate case where fuzziness is too high and newDirection is opposite to the normal
                if (otk::dot(newDirection, worldNormal) < 0.0f)
                {
                    newDirection = calculateReflectedDirection(rayDir, worldNormal);
                }
            }
            setRayOrigin(hitPoint);
            setRayDirection(newDirection);
            setCuRandStateMRG32k3a(curandState);
            setRayState(state);
            setStokesVector(make_float4(1.0f, 0.0f, 0.0f, 0.0f));
        }
        else
        {
            setRayOrigin(hitPoint);
            state.absorbed = 1;
            state.done = 1;
            setRayState(state);
        }
    }
    else
    {
        DBG_LOG_TEXT("Hit refractive or volume scattering material");
        float4 stokesVector = getStokesVector();
        float3 qMinusAxis = getQMinusAxis();

        unsigned int nextMedium;

        if (isFrontFace)
        {
            nextMedium = instanceId; // Current instance is the next medium
        }
        else
        {
            if (state.currentMediumHistorySize == 0)
            {
                // printf("Warning: Back face but no previous medium in history.\n");
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
#if !defined(NDEBUG)
                        if (state.currentMediumHistorySize - 2 < 0)
                        {
                            printf("Error: medium index %d out of bounds [0, 15]\n", state.currentMediumHistorySize - 2);
                            state.absorbed = 3;
                            state.done = 1;
                            setRayState(state);
                            return;
                        }
#endif
                        nextMedium = getMedium(state.currentMediumHistorySize - 2); // Get the medium one layer back
                    }
                }
                else
                {
                    nextMedium = state.currentMedium; // If not, two different media are overlapping, so stay in the current medium
                    state.numberOfWarnings++;
                    // printf("Warning: Two different media are overlapping, this leads to ill defined refractive indices of the overlap.\n");
                }
            }
        }
#if !defined(NDEBUG)
        if (nextMedium < 0 || nextMedium >= 15)
        {
            printf("Error: next medium index %d out of bounds [0, 15]\n", nextMedium);
            state.absorbed = 3;
            state.done = 1;
            setRayState(state);
            return;
        }
#endif
        MaterialType nextMaterialType = hgData->materials[nextMedium].type;
        RefractiveIndex nextRefractiveIndex;
        if (nextMaterialType == REFRACTIVE)
        {
            nextRefractiveIndex = hgData->materials[nextMedium].properties.refractive.refractiveIndex;
        }
        else if (nextMaterialType == VOLUME_SCATTERING)
        {
            nextRefractiveIndex = hgData->materials[nextMedium].properties.volumeScattering.refractiveIndex;
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
        rndSample = curand_uniform(&curandState);
        bool isReflected;
        if (params.useComplexFresnel)
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
                bool success = appendMedium(instanceId, state.currentMediumHistorySize);
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
                auto found = removeLastOccurence(instanceId, state.currentMediumHistorySize);
                if (!found)
                {
                    state.numberOfWarnings++;
                    DBG_LOG_TEXT("Warning: medium not found in history");
                }
                state.currentMedium = nextMedium;
            }
        }

        if (optixIsTriangleHit())
        {
            if (isReflected)
            {
                hitPoint = front;
            }
            else
            {
                hitPoint = back;
            }
        }

        setRayOrigin(hitPoint);
        setRayDirection(rayDir);
        setStokesVector(stokesVector);
        setQDirection(qMinusAxis);
        setCuRandStateMRG32k3a(curandState);
        setRayState(state);
    }
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