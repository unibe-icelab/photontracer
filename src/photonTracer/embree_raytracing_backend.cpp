// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <algorithm>
#include <atomic>
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
    SpawnPoint spawn;      // in world space
    unsigned int material; // of the innermost instance
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
    for (unsigned int level = 0; level < RTC_MAX_INSTANCE_LEVEL_COUNT; ++level)
    {
        rayHit.hit.instID[level] = RTC_INVALID_GEOMETRY_ID;
    }
    rtcIntersect1(structure.scene, &rayHit);

    TriangleHit result = {};
    if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
    {
        return result;
    }

    // Follow the instances of the hit, outermost first, down to the scene of the triangle
    const EmbreeAccelerationStructure *leaf = &structure;
    const EmbreeInstance *path[RTC_MAX_INSTANCE_LEVEL_COUNT];
    unsigned int depth = 0;
    result.material = 1; // a plain mesh uses material 1 for its inside
    while (depth < RTC_MAX_INSTANCE_LEVEL_COUNT && rayHit.hit.instID[depth] != RTC_INVALID_GEOMETRY_ID)
    {
        const EmbreeInstance &instance = leaf->instances[rayHit.hit.instID[depth]];
        path[depth++] = &instance;
        result.material = instance.materialId;
        leaf = instance.child;
    }

    const std::vector<float3> &vertices = leaf->mesh->getVertices();
    const unsigned int *triangle = &leaf->mesh->getIndices()[3 * static_cast<size_t>(rayHit.hit.primID)];
    result.found = true;
    result.distance = rayHit.ray.tfar;
    result.spawn = triangleSpawnPoint(vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]],
                                      rayHit.hit.u, rayHit.hit.v);
    for (unsigned int level = depth; level-- > 0;)
    {
        result.spawn = transformSpawnPoint(result.spawn, path[level]->transform);
    }
    // Front faces are those whose vertices run counter-clockwise as seen by the ray
    result.isFrontFace = otk::dot(result.spawn.normal, direction) < 0.0f;
    return result;
}

/// Counts a ray that left the scene in the HEALPix bin of its direction. The histogram of a pixel is
/// selected by the second and third launch index, as in the OptiX kernel.
void countEscapedDirection(const InputParameters &params, float3 direction, uint3 launchIndex, uint3 launchShape)
{
    static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free,
                  "the histogram is incremented through atomics");

    const int bin = healpixAng2PixRing(static_cast<int>(params.healpixNside), otk::normalize(direction));
    const uint32_t pixel = launchIndex.z * launchShape.y + launchIndex.y;
    const uint64_t index = static_cast<uint64_t>(pixel) * params.healpixBinCount + bin;
    if (bin >= 0 && index < static_cast<uint64_t>(params.healpixBinCount) * launchShape.y * launchShape.z)
    {
        auto *counter = reinterpret_cast<std::atomic<uint32_t> *>(params.deviceOutputBuffers.directionHistogramHealpix + index);
        counter->fetch_add(1, std::memory_order_relaxed);
    }
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
            if (params.outputFlags & OUT_DIRECTION_HISTOGRAM_HEALPIX)
            {
                countEscapedDirection(params, prd.direction, launchIndex, launchShape);
            }
            prd.state.done = true;
            break;
        }

        HitInfo hit;
        hit.rayOrigin = prd.origin;
        hit.rayDirection = prd.direction;
        hit.maxDistance = trace.distance;
        hit.instanceId = trace.material;
        hit.hitPoint = trace.spawn.position;
        hit.front = trace.spawn.front();
        hit.back = trace.spawn.back();
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
        origin = otk::dot(trace.spawn.normal, direction) > 0.0f ? trace.spawn.front() : trace.spawn.back();
    }
    return backFaces - frontFaces;
}

void requireSupportedOutputs(uint32_t outputFlags)
{
    if (outputFlags & (OUT_LOGS | OUT_LOG_OFFSETS))
    {
        throw std::runtime_error("The Embree backend does not support the LOGS and LOG_OFFSETS outputs yet");
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
    return RTC_MAX_INSTANCE_LEVEL_COUNT + 1; // the instances and the mesh
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
    auto structure = std::make_unique<EmbreeAccelerationStructure>(static_cast<const IRaytracingBackend *>(this));
    switch (geometry.getType())
    {
    case MESH:
        buildMesh(static_cast<const MeshGeometry &>(geometry), *structure);
        break;
    case MESH_INSTANCED:
        buildInstances(static_cast<const InstanceGeometry &>(geometry), *structure);
        break;
    default:
        throw std::runtime_error("Unknown geometry type");
    }
    checkDeviceError("build the geometry");
    geometry.setAccelerationStructure(std::move(structure));
}

void EmbreeRaytracingBackend::buildMesh(const MeshGeometry &mesh, EmbreeAccelerationStructure &structure)
{
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

    structure.mesh = &mesh;
    structure.scene = rtcNewScene(device_);
    // Watertight intersection; hits that slip through gaps would let rays leave a closed surface
    rtcSetSceneFlags(structure.scene, RTC_SCENE_FLAG_ROBUST | (mesh.isCompact() ? RTC_SCENE_FLAG_COMPACT : RTC_SCENE_FLAG_NONE));
    rtcSetSceneBuildQuality(structure.scene, RTC_BUILD_QUALITY_HIGH);
    rtcAttachGeometry(structure.scene, triangles);
    rtcReleaseGeometry(triangles);
    rtcCommitScene(structure.scene);
}

void EmbreeRaytracingBackend::buildInstances(const InstanceGeometry &instances, EmbreeAccelerationStructure &structure)
{
    const auto &subGeometries = instances.getSubGeometries();
    for (const auto &subGeometry : subGeometries)
    {
        if (!subGeometry->isBuiltBy(static_cast<const IRaytracingBackend *>(this)))
        {
            buildGeometry(*subGeometry);
        }
    }

    const std::vector<float> &matrices = instances.getInstanceTransforms();
    const size_t instanceCount = matrices.size() / 12;
    structure.instances.reserve(instanceCount);
    structure.scene = rtcNewScene(device_);
    rtcSetSceneFlags(structure.scene, RTC_SCENE_FLAG_ROBUST);
    rtcSetSceneBuildQuality(structure.scene, RTC_BUILD_QUALITY_HIGH);

    unsigned int deepestChild = 0;
    for (size_t i = 0; i < instanceCount; ++i)
    {
        const unsigned int particleTypeId = instances.getParticleTypeIds()[i];
        if (particleTypeId >= subGeometries.size())
        {
            throw std::runtime_error("Particle type ID out of bounds for subGeometries.");
        }
        const auto &child = *static_cast<const EmbreeAccelerationStructure *>(subGeometries[particleTypeId]->getAccelerationStructure());
        deepestChild = std::max(deepestChild, child.instanceLevels);

        EmbreeInstance instance;
        if (!makeInstanceTransform(&matrices[12 * i], instance.transform))
        {
            throw std::invalid_argument("The transform of instance " + std::to_string(i) + " is singular");
        }
        instance.materialId = instances.getMaterialIds()[i];
        instance.child = &child;
        structure.instances.push_back(instance);

        RTCGeometry geometry = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_INSTANCE);
        rtcSetGeometryInstancedScene(geometry, child.scene);
        rtcSetGeometryTransform(geometry, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, instance.transform.matrix);
        rtcCommitGeometry(geometry);
        const unsigned int id = rtcAttachGeometry(structure.scene, geometry);
        rtcReleaseGeometry(geometry);
        if (id != i)
        {
            throw std::runtime_error("Embree numbered the instances in another order");
        }
    }

    structure.instanceLevels = deepestChild + 1;
    if (structure.instanceLevels > RTC_MAX_INSTANCE_LEVEL_COUNT)
    {
        throw std::runtime_error("Instances are nested " + std::to_string(structure.instanceLevels) +
                                 " levels deep, but this build of the Embree backend supports " +
                                 std::to_string(RTC_MAX_INSTANCE_LEVEL_COUNT));
    }
    rtcCommitScene(structure.scene);
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
