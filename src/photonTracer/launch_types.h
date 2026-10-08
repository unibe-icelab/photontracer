// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include "portable_math.h"

enum RayGeneratorType
{
    RAYGEN_PARALLEL,
    RAYGEN_ISOTROPIC,
    RAYGEN_CAMERA,
};


union RayGeneratorData
{
    struct ParallelSource
    {
        uint32_t numberOfRays;
        float3 origin;
        float3 direction;
        float offsetRadius;
    } parallel;
    struct IsotropicSource
    {
        uint32_t numberOfRays;
        float3 center;
        float sourceRadius;
        float offsetRadius;
    } isotropic;
    struct CameraSource
    {
        float3 center;
        float3 pixel00;
        float3 pixelDeltaU;
        float3 pixelDeltaV;
        float3 defocusDiskU;
        float3 defocusDiskV;
        uint32_t imageWidth;
        uint32_t imageHeight;
        uint32_t samplesPerPixel;
        uint32_t enableDefocus;
    } camera;
};






enum class OutputType
{
    LAST_DIRECTION = 0,
    LAST_POSITION = 1,
    RAY_STATE = 2,
    LAST_MEDIUM_ID = 3,
    SCATTERING_COUNT = 4,
    NUMBER_OF_WARNINGS = 5,
    STOKES_VECTOR = 6,                // float4: [I, Q, U, V]
    STOKES_VECTOR_IN = 7,             // float4: [I, Q, U, V] at the source
    OPTICAL_PATH_LENGTH = 8,          // float: total distance traveled
    SOURCE_DIRECTION = 9,             // float3: initial ray direction
    SOURCE_POSITION = 10,             // float3: initial ray position
    SCATTERING_ANGLE = 11,            // float: angle between source direction and last direction
    Q_MINUS_AXIS_IN = 12,             // float3: Q- axis for the Stokes vector
    DIRECTION_HISTOGRAM_HEALPIX = 13, // uint32_t: histogram of scattered directions binned via HEALPix

    OUTPUT_TYPE_COUNT
};

enum OutputFlags : uint32_t
{
    OUT_LAST_DIRECTION = 1u << static_cast<uint32_t>(OutputType::LAST_DIRECTION),
    OUT_LAST_POSITION = 1u << static_cast<uint32_t>(OutputType::LAST_POSITION),
    OUT_RAY_STATE = 1u << static_cast<uint32_t>(OutputType::RAY_STATE),
    OUT_LAST_MEDIUM_ID = 1u << static_cast<uint32_t>(OutputType::LAST_MEDIUM_ID),
    OUT_SCATTERING_COUNT = 1u << static_cast<uint32_t>(OutputType::SCATTERING_COUNT),
    OUT_NUMBER_OF_WARNINGS = 1u << static_cast<uint32_t>(OutputType::NUMBER_OF_WARNINGS),
    OUT_STOKES_VECTOR = 1u << static_cast<uint32_t>(OutputType::STOKES_VECTOR),
    OUT_STOKES_VECTOR_IN = 1u << static_cast<uint32_t>(OutputType::STOKES_VECTOR_IN),
    OUT_OPTICAL_PATH_LENGTH = 1u << static_cast<uint32_t>(OutputType::OPTICAL_PATH_LENGTH),
    OUT_SOURCE_DIRECTION = 1u << static_cast<uint32_t>(OutputType::SOURCE_DIRECTION),
    OUT_SOURCE_POSITION = 1u << static_cast<uint32_t>(OutputType::SOURCE_POSITION),
    OUT_SCATTERING_ANGLE = 1u << static_cast<uint32_t>(OutputType::SCATTERING_ANGLE),
    OUT_Q_MINUS_AXIS_IN = 1u << static_cast<uint32_t>(OutputType::Q_MINUS_AXIS_IN),
    OUT_DIRECTION_HISTOGRAM_HEALPIX = 1u << static_cast<uint32_t>(OutputType::DIRECTION_HISTOGRAM_HEALPIX),
};

struct DeviceOutputBuffers
{
    float3 *lastDirection;               // Last direction of the ray
    float3 *lastPosition;                // Last position of the ray
    int *ray_state;                      // Absorbed flag for each ray
    int *lastMediumID;                   // Medium index for each ray
    uint32_t *numberOfWarnings;          // Number of warnings for each ray
    uint32_t *scatteringCount;          // Scattering count of each ray in the scene
    float4 *stokesVector;                // Stokes vector: [I, Q, U, V]
    float4 *stokesVectorIn;              // Stokes vector at the source: [I, Q, U, V]
    double *opticalPathLength;           // Total distance traveled by the ray
    float3 *sourceDirection;             // Initial ray direction
    float3 *sourcePosition;              // Initial ray position
    float *scatteringAngle;              // Angle between source direction and last direction
    float3 *qMinusAxisIn;                // Q- axis for the Stokes vector of the source
    uint32_t *directionHistogramHealpix; // Histogram of miss directions (HEALPix bins)
};

/// Index of a launch point in the output buffers
PT_INLINE PT_HD uint32_t linearizeLaunchIndex(uint3 index, uint3 dimensions)
{
    return index.x + index.y * dimensions.x + index.z * dimensions.x * dimensions.y;
}

constexpr uint32_t MAX_TRACED_RAYS = 256;

enum TraceEvent : uint32_t
{
    TRACE_INTERACTION,    ///< reflected, refracted or scattered, the ray goes on
    TRACE_ESCAPED,        ///< the ray left the scene
    TRACE_ABSORBED,       ///< the ray was absorbed
    TRACE_MAX_SCATTERING, ///< the ray reached the maximum scattering count
    TRACE_ERROR,          ///< the ray ended on an error
};

/// One step of a traced ray: from a start to the next interaction. Only 4-byte members, so the
/// layout is the same everywhere and numpy can read it.
struct TraceRecord
{
    uint32_t ray;
    uint32_t step;
    uint32_t event;
    uint32_t mediumIn;
    uint32_t mediumOut;
    float originIn[3];
    float directionIn[3];
    float originOut[3]; ///< where the ray continues, just off the surface
    float directionOut[3];
    float stokes[4];
    float opticalPathLength; ///< of the whole ray so far
};
static_assert(sizeof(TraceRecord) == 22 * sizeof(float), "TraceRecord must not have padding");

/// The rays to trace, sorted, and the buffers for their steps
struct TraceParams
{
    uint32_t rayCount;
    uint32_t maxSteps;
    uint32_t rays[MAX_TRACED_RAYS];
    TraceRecord *records; ///< [rayCount * maxSteps]
    uint32_t *stepCounts; ///< [rayCount], all steps including those beyond maxSteps
};
