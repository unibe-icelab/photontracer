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

Hit trace(RTCScene scene, float3 origin, float3 direction)
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
    rtcIntersect1(scene, &rayHit);
    return {rayHit.hit.geomID != RTC_INVALID_GEOMETRY_ID, rayHit.hit.primID, rayHit.ray.tfar, rayHit.hit.u, rayHit.hit.v};
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
        EXPECT_FALSE(trace(scene, entering ? spawn.front : spawn.back, reflected).found)
            << "scale " << scale << " distance " << distance << " axis aligned " << axisAligned << " stretch " << stretch.y << " " << stretch.z;

        // The transmitted ray goes on to another triangle
        const Hit second = trace(scene, entering ? spawn.back : spawn.front, direction);
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
