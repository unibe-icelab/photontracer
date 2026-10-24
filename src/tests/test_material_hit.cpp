// handleMaterialHit run on the CPU through MemoryRayContext, with scripted random numbers.

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>

#include "../photonTracer/material_hit.h"

namespace
{
// Returns the given numbers in order and counts how many were used
struct ScriptedSample
{
    std::vector<float> values;
    size_t used = 0;

    ScriptedSample(std::initializer_list<float> v) : values(v) {}
    float operator()() { return values.at(used++); }
};

Material refractive(float n, float k)
{
    Material m{};
    m.type = REFRACTIVE;
    m.properties.refractive.refractiveIndex = {n, k};
    return m;
}

Material diffuse(float albedo)
{
    Material m{};
    m.type = DIFFUSE;
    m.properties.diffuse.albedo = albedo;
    return m;
}

Material reflective(float reflectivity, float fuzziness)
{
    Material m{};
    m.type = REFLECTIVE;
    m.properties.reflective = {reflectivity, fuzziness};
    return m;
}

Material volumeScattering(float n, float k, float scattering, float g)
{
    Material m{};
    m.type = VOLUME_SCATTERING;
    m.properties.volumeScattering = {{n, k}, scattering, g};
    return m;
}

void expectNear(float3 a, float3 b, float tol = 1e-5f)
{
    EXPECT_NEAR(a.x, b.x, tol);
    EXPECT_NEAR(a.y, b.y, tol);
    EXPECT_NEAR(a.z, b.z, tol);
}

struct Scene
{
    HitGroupData hg{};
    RayData ray{};
    HitInfo hit{};

    // A ray falling straight down onto a horizontal surface at z = 0 from z = 5
    explicit Scene(Material inside)
    {
        hg.wavelengthUm = 1.0f;
        hg.materials[0] = refractive(1.0f, 0.0f);
        hg.materials[1] = inside;

        ray.direction = make_float3(0, 0, -1);
        ray.origin = make_float3(0, 0, 5);
        ray.stokesVector = make_float4(1, 0, 0, 0);
        ray.qMinusAxis = make_float3(1, 0, 0);

        hit.rayOrigin = ray.origin;
        hit.rayDirection = ray.direction;
        hit.maxDistance = 5.0f;
        hit.hitPoint = make_float3(0, 0, 0);
        hit.front = make_float3(0, 0, 0.001f);
        hit.back = make_float3(0, 0, -0.001f);
        hit.worldNormal = make_float3(0, 0, 1);
        hit.instanceId = 1;
        hit.isFrontFace = true;
    }

    template <typename Sampler>
    void run(Sampler &sample, bool complexFresnel = true)
    {
        MemoryRayContext ctx{&ray};
        handleMaterialHit(ctx, hit, hg, 1.0f, complexFresnel, sample);
    }
};
} // namespace

// With complex indices (the default) a sample at or below the reflectivity (4% here) reflects;
// with real indices a sample below the transmissivity transmits.

TEST(MaterialHit, EnterGlassTransmits)
{
    Scene scene(refractive(1.5f, 0.0f));
    ScriptedSample sample{0.5f, 0.5f};
    scene.run(sample);

    EXPECT_EQ(sample.used, 2u);
    EXPECT_EQ(scene.ray.state.currentMedium, 1u);
    EXPECT_EQ(scene.ray.state.currentMediumHistorySize, 1u);
    EXPECT_EQ(getMediumFromPacked(scene.ray.packedMediumHistory, 0), 1u);
    EXPECT_FALSE(scene.ray.state.done);
    expectNear(scene.ray.origin, scene.hit.back);
    expectNear(scene.ray.direction, make_float3(0, 0, -1));
    EXPECT_NEAR(scene.ray.opticalPathLength, 5.0, 1e-6); // 5 um in vacuum
}

TEST(MaterialHit, EnterGlassReflects)
{
    Scene scene(refractive(1.5f, 0.0f));
    ScriptedSample sample{0.5f, 0.0f};
    scene.run(sample);

    EXPECT_EQ(scene.ray.state.currentMedium, 0u);
    EXPECT_EQ(scene.ray.state.currentMediumHistorySize, 0u);
    expectNear(scene.ray.origin, scene.hit.front);
    expectNear(scene.ray.direction, make_float3(0, 0, 1));
}

TEST(MaterialHit, ExitGlassRestoresMedium)
{
    Scene scene(refractive(1.5f, 0.0f));
    scene.ray.state.currentMedium = 1;
    scene.ray.state.currentMediumHistorySize = 1;
    scene.ray.packedMediumHistory = 1;
    scene.hit.isFrontFace = false;

    ScriptedSample sample{0.5f, 0.5f};
    scene.run(sample);

    EXPECT_EQ(scene.ray.state.currentMedium, 0u);
    EXPECT_EQ(scene.ray.state.currentMediumHistorySize, 0u);
    EXPECT_EQ(scene.ray.packedMediumHistory, 0u);
    EXPECT_NEAR(scene.ray.opticalPathLength, 7.5, 1e-6); // 5 um at n = 1.5
}

TEST(MaterialHit, ExitIntoEnclosingMedium)
{
    // inside a droplet (material 2) inside glass (material 1); leaving the droplet returns to the glass
    Scene scene(refractive(1.5f, 0.0f));
    scene.hg.materials[2] = refractive(1.33f, 0.0f);
    scene.ray.state.currentMedium = 2;
    scene.ray.state.currentMediumHistorySize = 2;
    scene.ray.packedMediumHistory = 1u | (2u << 4);
    scene.hit.instanceId = 2;
    scene.hit.isFrontFace = false;

    ScriptedSample sample{0.5f, 0.5f};
    scene.run(sample);

    EXPECT_EQ(scene.ray.state.currentMedium, 1u);
    EXPECT_EQ(scene.ray.state.currentMediumHistorySize, 1u);
    EXPECT_EQ(scene.ray.packedMediumHistory, 1u);
}

TEST(MaterialHit, FullHistoryCountsAWarning)
{
    Scene scene(refractive(1.5f, 0.0f));
    scene.ray.state.currentMediumHistorySize = 8;
    scene.ray.packedMediumHistory = 0x11111111u;

    ScriptedSample sample{0.5f, 0.5f};
    scene.run(sample);

    EXPECT_EQ(scene.ray.state.numberOfWarnings, 1u);
    EXPECT_EQ(scene.ray.state.currentMedium, 1u);
    EXPECT_EQ(scene.ray.state.currentMediumHistorySize, 8u);
}

TEST(MaterialHit, AbsorptionInsideTheMedium)
{
    // alpha = 4 pi k / lambda = 0.126 / um, so a sample of 0.9 gives a free path of 0.84 um
    Scene scene(refractive(1.5f, 0.01f));
    scene.ray.state.currentMedium = 1;
    scene.ray.state.currentMediumHistorySize = 1;
    scene.ray.packedMediumHistory = 1;

    ScriptedSample sample{0.9f};
    scene.run(sample);

    EXPECT_EQ(sample.used, 1u);
    EXPECT_EQ(scene.ray.state.absorbed, 1u);
    EXPECT_TRUE(scene.ray.state.done);
    const float distance = -std::log(0.9f) / (4.0f * 3.14159265f * 0.01f);
    expectNear(scene.ray.origin, make_float3(0, 0, 5 - distance), 1e-4f);
    EXPECT_NEAR(scene.ray.opticalPathLength, 1.5 * distance, 1e-4);
}

TEST(MaterialHit, VolumeScatteringChangesDirection)
{
    Scene scene(volumeScattering(1.33f, 0.0f, 1.0f, 0.8f));
    scene.ray.state.currentMedium = 1;
    scene.ray.state.currentMediumHistorySize = 1;
    scene.ray.packedMediumHistory = 1;

    // free path, then the albedo decision (no absorption, so always a scatter), then the two angles
    ScriptedSample sample{0.9f, 0.5f, 0.25f, 0.75f};
    scene.run(sample);

    EXPECT_EQ(sample.used, 4u);
    EXPECT_FALSE(scene.ray.state.done);
    const float distance = -std::log(0.9f) / 1.0f;
    expectNear(scene.ray.origin, make_float3(0, 0, 5 - distance), 1e-4f);
    expectNear(scene.ray.direction, henyeyGreensteinDirection(make_float3(0, 0, -1), 0.8f, 0.25f, 0.75f));
    EXPECT_NEAR(otk::length(scene.ray.direction), 1.0f, 1e-5);
    EXPECT_NEAR(scene.ray.opticalPathLength, 1.33 * distance, 1e-4);
}

TEST(MaterialHit, DiffuseScatters)
{
    Scene scene(diffuse(0.5f));
    scene.ray.stokesVector = make_float4(1, 0.5f, 0, 0);

    ScriptedSample sample{0.5f, 0.3f, 0.25f, 0.75f};
    scene.run(sample);

    EXPECT_EQ(sample.used, 4u);
    EXPECT_FALSE(scene.ray.state.done);
    expectNear(scene.ray.origin, scene.hit.front);
    expectNear(scene.ray.direction, calculateLamberianDirection(0.25f, 0.75f, make_float3(0, 0, 1)));
    EXPECT_NEAR(scene.ray.direction.z, 0.5f, 1e-5);
    EXPECT_FLOAT_EQ(scene.ray.stokesVector.y, 0.0f); // depolarized
}

TEST(MaterialHit, DiffuseAbsorbs)
{
    Scene scene(diffuse(0.5f));
    ScriptedSample sample{0.5f, 0.9f};
    scene.run(sample);

    EXPECT_EQ(scene.ray.state.absorbed, 1u);
    EXPECT_TRUE(scene.ray.state.done);
    expectNear(scene.ray.origin, scene.hit.hitPoint);
}

TEST(MaterialHit, MirrorReflects)
{
    Scene scene(reflective(1.0f, 0.0f));
    ScriptedSample sample{0.5f, 0.5f};
    scene.run(sample);

    EXPECT_EQ(sample.used, 2u);
    expectNear(scene.ray.origin, scene.hit.front);
    expectNear(scene.ray.direction, make_float3(0, 0, 1));
}

TEST(MaterialHit, FuzzyMirrorStaysAboveTheSurface)
{
    Scene scene(reflective(1.0f, 0.5f));
    // the first point is outside the unit sphere and rejected, the second is used
    ScriptedSample sample{0.5f, 0.5f, 0.9f, 0.9f, 0.9f, 0.6f, 0.7f, 0.8f};
    scene.run(sample);

    EXPECT_EQ(sample.used, 8u);
    EXPECT_NEAR(otk::length(scene.ray.direction), 1.0f, 1e-5);
    EXPECT_GT(scene.ray.direction.z, 0.0f);
}

TEST(MaterialHit, RealFresnelMatchesComplexForTransparentMedia)
{
    Scene complexScene(refractive(1.5f, 0.0f));
    Scene realScene(refractive(1.5f, 0.0f));
    ScriptedSample a{0.5f, 0.3f}, b{0.5f, 0.3f};
    complexScene.run(a, true);
    realScene.run(b, false);

    expectNear(complexScene.ray.direction, realScene.ray.direction);
    EXPECT_EQ(complexScene.ray.state.currentMedium, realScene.ray.state.currentMedium);
    EXPECT_NEAR(complexScene.ray.stokesVector.x, realScene.ray.stokesVector.x, 1e-5);
}
