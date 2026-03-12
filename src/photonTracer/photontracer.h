// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <cuda_runtime.h>
#include <optix.h>
#include <vector>
#include <string>
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

enum RayGeneratorType
{
    RAYGEN_PARALLEL,
    RAYGEN_ISOTROPIC,
    RAYGEN_CAMERA,
};

enum MaterialType
{
    DIFFUSE,
    REFRACTIVE,
    VOLUME_SCATTERING,
    REFLECTIVE,
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

struct RefractiveIndex
{
    float r;
    float i;
};

union MaterialProperties
{
    struct Diffuse
    {
        float albedo; // Albedo for Diffuse material
    } diffuse;

    struct Refractive
    {
        RefractiveIndex refractiveIndex; // Refractive index for the material
    } refractive;

    struct VolumeScattering
    {
        RefractiveIndex refractiveIndex; // Refractive index for the medium (absorption coefficient in k)
        float scatteringCoefficient;     // Scattering coefficient for the medium
        float asymetryParameter;         // Asymmetry parameter for the Henyey-Greenstein phase function
    } volumeScattering;

    struct Reflective
    {
        float reflectivity; // Reflectivity for the material
        float fuzziness;    // Fuzziness for the material
    } reflective;
};

struct Material
{
    MaterialType type;             // Type of the material
    MaterialProperties properties; // Properties of the material
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
    LOGS = 13,                        // Logs or debug information
    LOG_OFFSETS = 14,                 // Offsets for logs
    DIRECTION_HISTOGRAM_HEALPIX = 15, // uint32_t: histogram of scattered directions binned via HEALPix

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
    OUT_LOGS = 1u << static_cast<uint32_t>(OutputType::LOGS),
    OUT_LOG_OFFSETS = 1u << static_cast<uint32_t>(OutputType::LOG_OFFSETS),
    OUT_DIRECTION_HISTOGRAM_HEALPIX = 1u << static_cast<uint32_t>(OutputType::DIRECTION_HISTOGRAM_HEALPIX),
};

constexpr uint32_t LOG_BYTES_PER_RAY = 1u << 20;

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
    char *logs;                          // Logs or debug information
    uint32_t *logOffsets;                // Offsets for logs
    uint32_t *directionHistogramHealpix; // Histogram of miss directions (HEALPix bins)
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

struct HitGroupData
{
    Material materials[16]; // Array of refractive indices for each material
    float wavelengthUm;
};

struct HitGroupDataDensity
{
    // No data needed
};