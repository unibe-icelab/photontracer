// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include "embree_raytracing_backend.h"
#include "light_trace.h"
#include "material_hit.h"
#include "pcg32_rng.h"
#include "ray_context.h"
#include "ray_generation.h"
#include "spawn_point.h"

namespace
{
constexpr float MAX_DISTANCE = 1e16f;

struct TriangleHit
{
    bool found;
    float distance;
    SpawnPoint spawn;
    bool isFrontFace;
};

TriangleHit intersect(const EmbreeAccelerationStructure &structure, float3 origin, float3 direction)
{
    RTCRayHit rayHit = {};
    rayHit.ray.org_x = origin.x;
    rayHit.ray.org_y = origin.y;
    rayHit.ray.org_z = origin.z;
    rayHit.ray.dir_x = direction.x;
    rayHit.ray.dir_y = direction.y;
    rayHit.ray.dir_z = direction.z;
    rayHit.ray.tnear = 0.0f;
    rayHit.ray.tfar = MAX_DISTANCE;
    rayHit.ray.mask = 0xFFFFFFFFu;
    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(structure.scene, &rayHit);

    TriangleHit result = {};
    if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
    {
        return result;
    }

    const std::vector<float3> &vertices = structure.mesh->getVertices();
    const unsigned int *triangle = &structure.mesh->getIndices()[3 * static_cast<size_t>(rayHit.hit.primID)];
    result.found = true;
    result.distance = rayHit.ray.tfar;
    result.spawn = triangleSpawnPoint(vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]],
                                      rayHit.hit.u, rayHit.hit.v);
    // Front faces are those whose vertices run counter-clockwise as seen by the ray
    result.isFrontFace = otk::dot(result.spawn.normal, direction) < 0.0f;
    return result;
}

void traceLightRay(const EmbreeAccelerationStructure &structure, const HitGroupData &hitGroup,
                   const InputParameters &params, uint3 launchIndex, uint3 launchShape)
{
    uint32_t idx = launchIndex.x + launchIndex.y * launchShape.x + launchIndex.z * launchShape.x * launchShape.y;

    Pcg32State rng = makePcg32(params.initSeed, idx);
    auto nextSample = [&rng]() { return rng.nextFloat(); };

    float3 rayOrigin, rayDirection;
    if (!computeRay(params.rayGeneratorType, params.rayGeneratorData, nextSample, launchIndex, idx, rayOrigin, rayDirection))
    {
        throw std::runtime_error("Unknown ray generator type");
    }

    RayData prd;
    float4 stokesIn;
    float3 qMinusAxisIn;
    initializeLightTraceRay(params.outputFlags, params.deviceOutputBuffers, idx, rayOrigin, rayDirection,
                            params.stokesVector, params.qMinusAxisSeed, nextSample, prd, stokesIn, qMinusAxisIn);

    MemoryRayContext ctx{&prd};
    uint32_t scatteringCount = 0;
    for (;;)
    {
        if (params.maxScatteringCount > 0 && scatteringCount >= params.maxScatteringCount)
        {
            prd.state.absorbed = 2;
            break;
        }

        const TriangleHit trace = intersect(structure, prd.origin, prd.direction);
        if (!trace.found)
        {
            prd.state.done = true;
            break;
        }

        HitInfo hit;
        hit.rayOrigin = prd.origin;
        hit.rayDirection = prd.direction;
        hit.maxDistance = trace.distance;
        hit.instanceId = 1; // a plain mesh uses material 1 for its inside
        hit.hitPoint = trace.spawn.position;
        hit.front = trace.spawn.front;
        hit.back = trace.spawn.back;
        hit.worldNormal = trace.spawn.normal;
        hit.isFrontFace = trace.isFrontFace;
        if (otk::dot(hit.worldNormal, hit.rayDirection) > 0.0f)
        {
            hit.worldNormal = -hit.worldNormal;
            std::swap(hit.front, hit.back);
        }

        handleMaterialHit(ctx, hit, hitGroup, params.lengthScale, params.useComplexFresnel, nextSample);

        if (prd.state.done)
        {
            break;
        }
        scatteringCount++;
    }

    writeMainTraceOutputs(params.outputFlags, params.deviceOutputBuffers, idx, prd,
                          rayDirection, stokesIn, qMinusAxisIn, scatteringCount);
}

/// Surfaces crossed by a ray along +z from a random point of the box: leaving minus entering
int32_t countSurfaceCrossings(const EmbreeAccelerationStructure &structure, const InputParametersSampleDensity &params, uint32_t idx)
{
    Pcg32State rng = makePcg32(params.initSeed, idx);
    auto nextSample = [&rng]() { return rng.nextFloat(); };

    float3 origin = computeDensitySampleOrigin(params.boxMin, params.boxMax, nextSample);
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);

    int32_t frontFaces = 0;
    int32_t backFaces = 0;
    for (;;)
    {
        const TriangleHit trace = intersect(structure, origin, direction);
        if (!trace.found)
        {
            break;
        }
        (trace.isFrontFace ? frontFaces : backFaces)++;
        // Continue straight on the far side of the surface
        origin = otk::dot(trace.spawn.normal, direction) > 0.0f ? trace.spawn.front : trace.spawn.back;
    }
    return backFaces - frontFaces;
}

void requireSupportedOutputs(uint32_t outputFlags)
{
    if (outputFlags & (OUT_LOGS | OUT_LOG_OFFSETS | OUT_DIRECTION_HISTOGRAM_HEALPIX))
    {
        throw std::runtime_error("The Embree backend does not support the LOGS, LOG_OFFSETS and DIRECTION_HISTOGRAM_HEALPIX outputs yet");
    }
}
} // namespace

EmbreeRaytracingBackend::EmbreeRaytracingBackend(uint32_t cpuThreads)
    : cpuThreads_(cpuThreads)
{
    const std::string config = cpuThreads > 0 ? "threads=" + std::to_string(cpuThreads) : "";
    device_ = rtcNewDevice(config.c_str());
    if (!device_)
    {
        throw std::runtime_error("Could not create the Embree device: error " + std::to_string(rtcGetDeviceError(nullptr)));
    }
}

EmbreeRaytracingBackend::~EmbreeRaytracingBackend()
{
    rtcReleaseDevice(device_);
}

void EmbreeRaytracingBackend::checkDeviceError(const char *what) const
{
    const RTCError error = rtcGetDeviceError(device_);
    if (error != RTC_ERROR_NONE)
    {
        throw std::runtime_error(std::string("Embree failed to ") + what + ": error " + std::to_string(error));
    }
}

uint32_t EmbreeRaytracingBackend::getMaxTraversableGraphDepth() const
{
    return 1;
}

uint32_t EmbreeRaytracingBackend::getMaxSubGeometries() const
{
    return std::numeric_limits<uint32_t>::max();
}

uint32_t EmbreeRaytracingBackend::getMaxMeshTriangles() const
{
    return std::numeric_limits<uint32_t>::max();
}

void *EmbreeRaytracingBackend::allocateBuffer(size_t bytes)
{
    void *buffer = std::calloc(std::max<size_t>(bytes, 1), 1);
    if (!buffer)
    {
        throw std::bad_alloc();
    }
    return buffer;
}

void EmbreeRaytracingBackend::freeBuffer(void *buffer) noexcept
{
    std::free(buffer);
}

void EmbreeRaytracingBackend::clearBuffer(void *buffer, size_t bytes)
{
    std::memset(buffer, 0, bytes);
}

void EmbreeRaytracingBackend::copyBufferToHost(void *host, const void *buffer, size_t bytes)
{
    std::memcpy(host, buffer, bytes);
}

void EmbreeRaytracingBackend::buildGeometry(IGeometry &geometry)
{
    if (geometry.getType() != MESH)
    {
        throw std::runtime_error("The Embree backend only supports mesh geometries so far");
    }
    const MeshGeometry &mesh = static_cast<const MeshGeometry &>(geometry);

    RTCGeometry triangles = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);
    const size_t vertexCount = mesh.getVertices().size();
    const size_t triangleCount = mesh.getIndices().size() / 3;
    static_assert(sizeof(float3) == 3 * sizeof(float), "vertices are copied as packed floats");

    void *vertices = rtcSetNewGeometryBuffer(triangles, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                             sizeof(float3), vertexCount);
    void *indices = rtcSetNewGeometryBuffer(triangles, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                                            3 * sizeof(unsigned int), triangleCount);
    if (!vertices || !indices)
    {
        rtcReleaseGeometry(triangles);
        checkDeviceError("allocate the mesh");
        throw std::bad_alloc();
    }
    std::memcpy(vertices, mesh.getVertices().data(), vertexCount * sizeof(float3));
    std::memcpy(indices, mesh.getIndices().data(), triangleCount * 3 * sizeof(unsigned int));
    rtcCommitGeometry(triangles);

    auto structure = std::make_unique<EmbreeAccelerationStructure>(static_cast<const IRaytracingBackend *>(this));
    structure->mesh = &mesh;
    structure->scene = rtcNewScene(device_);
    // Watertight intersection; hits that slip through gaps would let rays leave a closed surface
    rtcSetSceneFlags(structure->scene, RTC_SCENE_FLAG_ROBUST | (mesh.isCompact() ? RTC_SCENE_FLAG_COMPACT : RTC_SCENE_FLAG_NONE));
    rtcSetSceneBuildQuality(structure->scene, RTC_BUILD_QUALITY_HIGH);
    rtcAttachGeometry(structure->scene, triangles);
    rtcReleaseGeometry(triangles);
    rtcCommitScene(structure->scene);
    checkDeviceError("build the mesh");

    geometry.setAccelerationStructure(std::move(structure));
}

const EmbreeAccelerationStructure &EmbreeRaytracingBackend::accelerationStructure(const IGeometry &geometry) const
{
    if (!geometry.isBuiltBy(static_cast<const IRaytracingBackend *>(this)))
    {
        throw std::runtime_error("The geometry has not been built by this backend");
    }
    return *static_cast<const EmbreeAccelerationStructure *>(geometry.getAccelerationStructure());
}

void EmbreeRaytracingBackend::initializePipeline(const std::vector<Material> &materials, float wavelengthUm, uint32_t,
                                                 RandomNumberGenerator generator)
{
    if (!supportsRandomNumberGenerator(generator))
    {
        throw std::invalid_argument("The Embree backend only supports the PCG32 random number generator");
    }
    updateShaderBindingTable(materials, wavelengthUm);
    hasPipeline_ = true;
}

void EmbreeRaytracingBackend::updateShaderBindingTable(const std::vector<Material> &materials, float wavelengthUm)
{
    std::copy(materials.begin(), materials.end(), hitGroup_.materials);
    hitGroup_.wavelengthUm = wavelengthUm;
}

void EmbreeRaytracingBackend::launch(InputParameters &params, const IGeometry &geometry, uint3 launchShape)
{
    requireSupportedOutputs(params.outputFlags);
    const EmbreeAccelerationStructure &structure = accelerationStructure(geometry);

    const uint64_t rayCount = static_cast<uint64_t>(launchShape.x) * launchShape.y * launchShape.z;
    tbb::task_arena arena(cpuThreads_ > 0 ? static_cast<int>(cpuThreads_) : tbb::task_arena::automatic);
    arena.execute([&]
                  { tbb::parallel_for(tbb::blocked_range<uint64_t>(0, rayCount, 256), [&](const tbb::blocked_range<uint64_t> &range)
                                      {
                                          for (uint64_t i = range.begin(); i != range.end(); ++i)
                                          {
                                              const uint32_t x = static_cast<uint32_t>(i % launchShape.x);
                                              const uint32_t y = static_cast<uint32_t>(i / launchShape.x % launchShape.y);
                                              const uint32_t z = static_cast<uint32_t>(i / launchShape.x / launchShape.y);
                                              traceLightRay(structure, hitGroup_, params, make_uint3(x, y, z), launchShape);
                                          } }); });
}

void EmbreeRaytracingBackend::initializeDensityPipeline(uint32_t, RandomNumberGenerator generator)
{
    if (!supportsRandomNumberGenerator(generator))
    {
        throw std::invalid_argument("The Embree backend only supports the PCG32 random number generator");
    }
    hasDensityPipeline_ = true;
}

void EmbreeRaytracingBackend::launchDensity(InputParametersSampleDensity &params, const IGeometry &geometry)
{
    const EmbreeAccelerationStructure &structure = accelerationStructure(geometry);

    tbb::task_arena arena(cpuThreads_ > 0 ? static_cast<int>(cpuThreads_) : tbb::task_arena::automatic);
    arena.execute([&]
                  { tbb::parallel_for(tbb::blocked_range<uint32_t>(0, params.numberOfRays, 256), [&](const tbb::blocked_range<uint32_t> &range)
                                      {
                                          for (uint32_t i = range.begin(); i != range.end(); ++i)
                                          {
                                              params.intersectionCountBuffer[i] = countSurfaceCrossings(structure, params, i);
                                          } }); });
}
