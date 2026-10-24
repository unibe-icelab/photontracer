#include <gtest/gtest.h>

#include <embree4/rtcore.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

#include "spawn_point.h"

namespace
{
struct Mesh
{
    std::vector<float3> vertices;
    std::vector<unsigned int> indices;
};

// A cube of edge length 2 whose faces are split into n x n quads
Mesh cube(int n)
{
    Mesh mesh;
    for (int axis = 0; axis < 3; ++axis)
    {
        for (int sign = -1; sign <= 1; sign += 2)
        {
            const unsigned int first = static_cast<unsigned int>(mesh.vertices.size());
            for (int j = 0; j <= n; ++j)
            {
                for (int i = 0; i <= n; ++i)
                {
                    const float a = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(n);
                    const float b = -1.0f + 2.0f * static_cast<float>(j) / static_cast<float>(n);
                    const float c[3] = {a, b, static_cast<float>(sign)};
                    mesh.vertices.push_back(make_float3(c[(3 - axis) % 3], c[(4 - axis) % 3], c[(5 - axis) % 3]));
                }
            }
            for (int j = 0; j < n; ++j)
            {
                for (int i = 0; i < n; ++i)
                {
                    const unsigned int v00 = first + j * (n + 1) + i;
                    const unsigned int v10 = v00 + 1;
                    const unsigned int v01 = v00 + (n + 1);
                    const unsigned int v11 = v01 + 1;
                    const bool flip = sign < 0;
                    for (const auto &t : {std::array<unsigned int, 3>{v00, v10, v11}, std::array<unsigned int, 3>{v00, v11, v01}})
                    {
                        mesh.indices.push_back(t[0]);
                        mesh.indices.push_back(flip ? t[2] : t[1]);
                        mesh.indices.push_back(flip ? t[1] : t[2]);
                    }
                }
            }
        }
    }
    return mesh;
}

float3 rotate(const float m[9], float3 v)
{
    return make_float3(m[0] * v.x + m[1] * v.y + m[2] * v.z, m[3] * v.x + m[4] * v.y + m[5] * v.z,
                       m[6] * v.x + m[7] * v.y + m[8] * v.z);
}

// Random rotation from a normalized quaternion
void randomRotation(std::mt19937 &gen, float m[9])
{
    std::normal_distribution<float> normal;
    float q[4] = {normal(gen), normal(gen), normal(gen), normal(gen)};
    const float norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (float &x : q)
    {
        x /= norm;
    }
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    const float r[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
    std::copy(r, r + 9, m);
}

struct Hit
{
    bool found;
    unsigned int primitive;
    float distance;
    float u, v;
};

struct InstanceHit
{
    Hit hit;
    unsigned int instances[RTC_MAX_INSTANCE_LEVEL_COUNT];
};

Hit trace(RTCScene scene, float3 origin, float3 direction, InstanceHit *instanceHit = nullptr)
{
    RTCRayHit rayHit = {};
    rayHit.ray.org_x = origin.x;
    rayHit.ray.org_y = origin.y;
    rayHit.ray.org_z = origin.z;
    rayHit.ray.dir_x = direction.x;
    rayHit.ray.dir_y = direction.y;
    rayHit.ray.dir_z = direction.z;
    rayHit.ray.tfar = 1e16f;
    rayHit.ray.mask = 0xFFFFFFFFu;
    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    for (unsigned int level = 0; level < RTC_MAX_INSTANCE_LEVEL_COUNT; ++level)
    {
        rayHit.hit.instID[level] = RTC_INVALID_GEOMETRY_ID;
    }
    rtcIntersect1(scene, &rayHit);
    const Hit hit = {rayHit.hit.geomID != RTC_INVALID_GEOMETRY_ID, rayHit.hit.primID, rayHit.ray.tfar, rayHit.hit.u, rayHit.hit.v};
    if (instanceHit)
    {
        instanceHit->hit = hit;
        std::copy(rayHit.hit.instID, rayHit.hit.instID + RTC_MAX_INSTANCE_LEVEL_COUNT, instanceHit->instances);
    }
    return hit;
}

RTCScene meshScene(RTCDevice device, const Mesh &mesh)
{
    RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    float *vertices = static_cast<float *>(rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                                                   sizeof(float3), mesh.vertices.size()));
    unsigned int *indices = static_cast<unsigned int *>(rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                                                                                3 * sizeof(unsigned int), mesh.indices.size() / 3));
    std::copy(&mesh.vertices[0].x, &mesh.vertices[0].x + 3 * mesh.vertices.size(), vertices);
    std::copy(mesh.indices.begin(), mesh.indices.end(), indices);
    rtcCommitGeometry(geometry);
    RTCScene scene = rtcNewScene(device);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
    rtcAttachGeometry(scene, geometry);
    rtcReleaseGeometry(geometry);
    rtcCommitScene(scene);
    return scene;
}

float3 applyTransform(const InstanceTransform &transform, float3 p)
{
    const float *m = transform.matrix;
    return make_float3(m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
                       m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
}
// Traces random rays at a cube and spawns from the first hit. Returns the number of rays that hit.
int checkSpawnPoints(RTCDevice device, std::mt19937 &gen, float scale, float distance, float3 stretch, bool axisAligned)
{
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

    Mesh mesh = cube(8);
    float rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (!axisAligned)
    {
        randomRotation(gen, rotation);
    }
    const float3 shift = make_float3(distance, -0.7f * distance, 0.3f * distance) * scale;
    auto place = [&](float3 p) { return rotate(rotation, make_float3(p.x * stretch.x, p.y * stretch.y, p.z * stretch.z)) * scale + shift; };
    for (float3 &vertex : mesh.vertices)
    {
        vertex = place(vertex);
    }

    RTCScene scene = meshScene(device, mesh);

    int hits = 0;
    for (int i = 0; i < 20000; ++i)
    {
        // from outside, aimed at a random point of the cube
        const float3 target = place(make_float3(uniform(gen), uniform(gen), uniform(gen)));
        const float3 direction = otk::normalize(make_float3(uniform(gen), uniform(gen), uniform(gen)));
        const Hit first = trace(scene, target - direction * (4.0f * scale * stretch.z), direction);
        if (!first.found)
        {
            continue;
        }
        ++hits;
        const unsigned int *t = &mesh.indices[3 * first.primitive];
        const SpawnPoint spawn = triangleSpawnPoint(mesh.vertices[t[0]], mesh.vertices[t[1]], mesh.vertices[t[2]], first.u, first.v);
        const bool entering = otk::dot(direction, spawn.normal) < 0.0f;

        // The reflected ray leaves a convex mesh from the outside, so it must not hit anything
        const float3 reflected = direction - spawn.normal * (2.0f * otk::dot(direction, spawn.normal));
        EXPECT_FALSE(trace(scene, entering ? spawn.front() : spawn.back(), reflected).found)
            << "scale " << scale << " distance " << distance << " axis aligned " << axisAligned << " stretch " << stretch.y << " " << stretch.z;

        // The transmitted ray goes on to another triangle
        const Hit second = trace(scene, entering ? spawn.back() : spawn.front(), direction);
        EXPECT_TRUE(second.found && second.primitive != first.primitive)
            << "scale " << scale << " distance " << distance << " axis aligned " << axisAligned << " stretch " << stretch.y << " " << stretch.z;
    }
    rtcReleaseScene(scene);
    return hits;
}
} // namespace

// Spawning from the offset points must never hit the triangle the ray just left: for meshes
// far from the origin, small or large, slivers, planes through the origin, and any ray angle.
TEST(SpawnPoint, DoesNotHitTheTriangleItLeft)
{
    RTCDevice device = rtcNewDevice(nullptr);
    std::mt19937 gen(1234);

    const float scales[] = {1e-3f, 1.0f, 1e3f};
    const float distances[] = {0.0f, 10.0f, 1e3f}; // of the mesh from the origin, in mesh sizes
    const float3 stretches[] = {make_float3(1, 1, 1), make_float3(1, 1, 30), make_float3(1, 0.03f, 30)};

    int hits = 0;
    for (float3 stretch : stretches)
    {
        for (float scale : scales)
        {
            for (float distance : distances)
            {
                for (bool axisAligned : {false, true})
                {
                    hits += checkSpawnPoints(device, gen, scale, distance, stretch, axisAligned);
                }
            }
        }
    }
    EXPECT_GT(hits, 300000);
    rtcReleaseDevice(device);
}

// Random rotation, stretch, scale and shift as the 3x4 matrix of an instance
InstanceTransform randomTransform(std::mt19937 &gen, float scale, float distance, float3 stretch)
{
    float rotation[9];
    randomRotation(gen, rotation);
    float matrix[12];
    for (int row = 0; row < 3; ++row)
    {
        const float s[3] = {stretch.x, stretch.y, stretch.z};
        for (int column = 0; column < 3; ++column)
        {
            matrix[4 * row + column] = rotation[3 * row + column] * s[column] * scale;
        }
    }
    matrix[3] = distance * scale;
    matrix[7] = -0.7f * distance * scale;
    matrix[11] = 0.3f * distance * scale;
    InstanceTransform transform;
    EXPECT_TRUE(makeInstanceTransform(matrix, transform));
    return transform;
}

// A cube placed in the world by `levels` nested instances. Returns the number of rays that hit.
int checkNestedSpawnPoints(RTCDevice device, std::mt19937 &gen, int levels, float scale, float distance, float3 stretch)
{
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    const Mesh cubeMesh = cube(8);

    // transforms[0] is the innermost instance
    std::vector<InstanceTransform> transforms;
    std::vector<RTCScene> scenes = {meshScene(device, cubeMesh)};
    for (int level = 0; level < levels; ++level)
    {
        transforms.push_back(randomTransform(gen, level == 0 ? scale : 1.0f, level == 0 ? distance : 1.0f, level == 0 ? stretch : make_float3(1, 1, 1)));
        RTCGeometry instance = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
        rtcSetGeometryInstancedScene(instance, scenes.back());
        rtcSetGeometryTransform(instance, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, transforms.back().matrix);
        rtcCommitGeometry(instance);
        RTCScene scene = rtcNewScene(device);
        rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
        rtcAttachGeometry(scene, instance);
        rtcReleaseGeometry(instance);
        rtcCommitScene(scene);
        scenes.push_back(scene);
    }
    auto toWorld = [&](float3 p)
    {
        for (const InstanceTransform &transform : transforms)
        {
            p = applyTransform(transform, p);
        }
        return p;
    };
    // The cube is at most this far from its center in the world
    float radius = 0.0f;
    for (float3 corner : {make_float3(1, 1, 1), make_float3(-1, 1, 1), make_float3(1, -1, 1), make_float3(1, 1, -1)})
    {
        const float3 d = toWorld(corner) - toWorld(make_float3(0, 0, 0));
        radius = std::fmax(radius, otk::length(d));
    }

    int hits = 0;
    for (int i = 0; i < 20000; ++i)
    {
        const float3 target = toWorld(make_float3(uniform(gen), uniform(gen), uniform(gen)));
        const float3 direction = otk::normalize(make_float3(uniform(gen), uniform(gen), uniform(gen)));
        InstanceHit first;
        trace(scenes.back(), target - direction * (3.0f * radius), direction, &first);
        if (!first.hit.found)
        {
            continue;
        }
        ++hits;

        const unsigned int *t = &cubeMesh.indices[3 * first.hit.primitive];
        SpawnPoint spawn = triangleSpawnPoint(cubeMesh.vertices[t[0]], cubeMesh.vertices[t[1]], cubeMesh.vertices[t[2]], first.hit.u, first.hit.v);
        for (const InstanceTransform &transform : transforms)
        {
            spawn = transformSpawnPoint(spawn, transform);
        }
        const bool entering = otk::dot(direction, spawn.normal) < 0.0f;

        const float3 reflected = direction - spawn.normal * (2.0f * otk::dot(direction, spawn.normal));
        const Hit again = trace(scenes.back(), entering ? spawn.front() : spawn.back(), reflected);
        EXPECT_FALSE(again.found) << "levels " << levels << " scale " << scale << " distance " << distance << " stretch " << stretch.y << " " << stretch.z;

        const Hit second = trace(scenes.back(), entering ? spawn.back() : spawn.front(), direction);
        EXPECT_TRUE(second.found && second.primitive != first.hit.primitive)
            << "levels " << levels << " scale " << scale << " distance " << distance << " stretch " << stretch.y << " " << stretch.z;
    }
    for (RTCScene scene : scenes)
    {
        rtcReleaseScene(scene);
    }
    return hits;
}

// The same as above for meshes that are placed by one or more instance transforms
TEST(SpawnPoint, DoesNotHitTheTriangleItLeftThroughInstances)
{
    RTCDevice device = rtcNewDevice(nullptr);
    std::mt19937 gen(4321);

    int hits = 0;
    for (int levels : {1, 2})
    {
        for (float scale : {1e-3f, 1.0f, 1e3f})
        {
            for (float distance : {0.0f, 10.0f, 1e3f})
            {
                for (float3 stretch : {make_float3(1, 1, 1), make_float3(1, 0.03f, 30)})
                {
                    hits += checkNestedSpawnPoints(device, gen, levels, scale, distance, stretch);
                }
            }
        }
    }
    EXPECT_GT(hits, 200000);
    rtcReleaseDevice(device);
}
