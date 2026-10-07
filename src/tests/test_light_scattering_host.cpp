// CPU-only counterpart of test_light_scattering.cu: the same physics functions
// called directly, built without CUDA.

#include <array>
#include <gtest/gtest.h>

#include "../photonTracer/light_scattering.h"

namespace
{
constexpr float kPi = 3.14159265358979323846f;

float rad(float degrees) { return degrees * kPi / 180.0f; }

void expectNear(float3 a, float3 b, float tol)
{
    EXPECT_NEAR(a.x, b.x, tol);
    EXPECT_NEAR(a.y, b.y, tol);
    EXPECT_NEAR(a.z, b.z, tol);
}

void expectNear(float4 a, float4 b, float tol)
{
    EXPECT_NEAR(a.x, b.x, tol);
    EXPECT_NEAR(a.y, b.y, tol);
    EXPECT_NEAR(a.z, b.z, tol);
    EXPECT_NEAR(a.w, b.w, tol);
}

void expectNear(const otk::Transform4 &a, const otk::Transform4 &b, float tol)
{
    for (int row = 0; row < 4; ++row)
    {
        expectNear(a.m[row], b.m[row], tol);
    }
}
} // namespace

TEST(LightScattering, Absorption)
{
    const float maxDistance = 1e-1f;
    const float wavelengthUm = 0.55f;
    const float lengthScale = 1e-3f;
    float k = 1.96e-9f;
    float travelled = 0.0f;

    EXPECT_FALSE(calculateAbsorption(maxDistance, wavelengthUm, lengthScale, k, 0.5f, travelled));
    EXPECT_NEAR(travelled, maxDistance, 1e-4);

    EXPECT_TRUE(calculateAbsorption(maxDistance, wavelengthUm, lengthScale, k, 0.999999f, travelled));
    EXPECT_NEAR(travelled, 0.02233042f, 1e-3);

    k = 0.0f;
    EXPECT_FALSE(calculateAbsorption(maxDistance, wavelengthUm, lengthScale, k, 0.5f, travelled));
}

TEST(LightScattering, SnellsLaw)
{
    float cosTheta, cosThetaPrime;
    bool tir;

    // ice -> air at 50 degrees: total internal reflection
    calculateSnellsLaw(otk::normalize(make_float3(0.76604444311f, -0.64278760968f, 0.0f)),
                       make_float3(0, 1, 0), 1.31f, 1.0f, cosTheta, cosThetaPrime, tir);
    EXPECT_NEAR(cosTheta, 0.64278758f, 1e-4);
    EXPECT_TRUE(tir);
    EXPECT_FLOAT_EQ(cosThetaPrime, 0.0f);

    // ice -> air at 45 degrees
    calculateSnellsLaw(otk::normalize(make_float3(1, -1, 0)), make_float3(0, 1, 0), 1.31f, 1.0f, cosTheta, cosThetaPrime, tir);
    EXPECT_NEAR(cosTheta, 0.70710677f, 1e-4);
    EXPECT_NEAR(cosThetaPrime, 0.37676269f, 1e-4);
    EXPECT_FALSE(tir);

    // air -> ice at 45 degrees
    calculateSnellsLaw(otk::normalize(make_float3(-1, -1, 0)), make_float3(0, 1, 0), 1.0f, 1.31f, cosTheta, cosThetaPrime, tir);
    EXPECT_NEAR(cosTheta, 0.70710677f, 1e-4);
    EXPECT_NEAR(cosThetaPrime, 0.84180862f, 1e-4);
    EXPECT_FALSE(tir);
}

TEST(LightScattering, RefractedDirection)
{
    const float3 normal = make_float3(0, 1, 0);
    const float3 down = otk::normalize(make_float3(1, -1, 0));

    // air -> ice
    expectNear(calculateRefractedDirection(down, normal, std::cos(rad(45.0f)), std::cos(rad(32.3905f)), 1.0f, 1.32f),
               make_float3(0.5357f, -0.8444f, 0.0f), 1e-4);

    // ice -> air
    expectNear(calculateRefractedDirection(down, normal, std::cos(rad(45.0f)), std::cos(rad(68.968f)), 1.32f, 1.0f),
               make_float3(0.93338f, -0.35889f, 0.0f), 1e-4);

    // ice -> air, normal pointing along -z
    expectNear(calculateRefractedDirection(make_float3(0.530861f, 0.0f, 0.847459f), make_float3(0, 0, -1),
                                           0.847459f, 0.707107f, 1.332f, 1.0f),
               make_float3(0.707107f, 0.0f, 0.707107f), 1e-4);
}

TEST(LightScattering, ReflectedDirection)
{
    const float3 normal = make_float3(0, 1, 0);
    expectNear(calculateReflectedDirection(otk::normalize(make_float3(1, -1, 0)), normal),
               otk::normalize(make_float3(1, 1, 0)), 1e-4);
    expectNear(calculateReflectedDirection(otk::normalize(make_float3(100, -1, 100)), normal),
               otk::normalize(make_float3(100, 1, 100)), 1e-4);
}

TEST(LightScattering, SignedRotationAboutAxis)
{
    const float3 axis = make_float3(0, 0, 1);
    const float3 x = make_float3(1, 0, 0);
    EXPECT_NEAR(signedRotationAboutAxis(axis, x, make_float3(0, 1, 0)), kPi / 2, 1e-5);
    EXPECT_NEAR(signedRotationAboutAxis(axis, x, make_float3(0, -1, 0)), -kPi / 2, 1e-5);
    EXPECT_NEAR(signedRotationAboutAxis(-axis, x, make_float3(0, 1, 0)), -kPi / 2, 1e-5);
}

TEST(LightScattering, ScatteringPlaneAxes)
{
    const float3 k = otk::normalize(make_float3(0.2f, -1.0f, 0.3f));
    const float3 surfaceNormal = make_float3(0, 1, 0);
    float3 axis;
    scatteringPlaneNormalAxis(k, surfaceNormal, axis);

    expectNear(axis, otk::normalize(otk::cross(surfaceNormal, k)), 1e-6);
    EXPECT_NEAR(otk::dot(axis, k), 0.0f, 1e-5);
    EXPECT_NEAR(otk::dot(axis, surfaceNormal), 0.0f, 1e-5);

    // k parallel to the normal has no scattering plane
    scatteringPlaneNormalAxis(surfaceNormal, surfaceNormal, axis);
    EXPECT_TRUE(std::isnan(axis.x));
}

TEST(LightScattering, LambertianDirection)
{
    const float3 normal = make_float3(0, 1, 0);
    const float theta = 2.0f * kPi * 0.25f;
    const float phi = std::acos(2.0f * 0.75f - 1.0f);
    const float3 expected = otk::normalize(normal + make_float3(std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta), std::cos(phi)));

    const float3 direction = calculateLamberianDirection(0.25f, 0.75f, normal);
    expectNear(direction, expected, 1e-5);
    EXPECT_NEAR(otk::length(direction), 1.0f, 1e-5);
    EXPECT_GT(otk::dot(direction, normal), 0.0f);
}

TEST(LightScattering, ReflectivitySchlick)
{
    const float na = 1.33f, nb = 1.0f;
    const float r0 = std::pow((na - nb) / (na + nb), 2.0f);
    EXPECT_NEAR(calculateReflectivitySchlick(1.0f, na, nb), r0, 1e-6);
    EXPECT_NEAR(calculateReflectivitySchlick(0.25f, na, nb), r0 + (1.0f - r0) * std::pow(0.75f, 5.0f), 1e-6);
}

TEST(LightScattering, RotationMatrix)
{
    const float c = 0.5f, s = 0.86603f; // cos and sin of 2 * 30 degrees
    otk::Transform4 expected;
    expected.m[0] = make_float4(1, 0, 0, 0);
    expected.m[1] = make_float4(0, c, s, 0);
    expected.m[2] = make_float4(0, -s, c, 0);
    expected.m[3] = make_float4(0, 0, 0, 1);
    expectNear(calculateRotationMatrix(rad(30.0f)), expected, 1e-4);
}

TEST(LightScattering, MullerTotalReflection)
{
    const otk::Transform4 m = calculateMullerTotalReflection(std::cos(rad(60.0f)), 1.5f, 1.0f);
    const float c = 0.76086956f, s = -0.64890486f;
    EXPECT_NEAR(m.m[0].x, 1.0f, 1e-6);
    EXPECT_NEAR(m.m[1].y, 1.0f, 1e-6);
    EXPECT_NEAR(m.m[1].x, 0.0f, 1e-6);
    EXPECT_NEAR(m.m[2].z, c, 1e-5);
    EXPECT_NEAR(m.m[2].w, s, 1e-5);
    EXPECT_NEAR(m.m[3].z, -s, 1e-5);
    EXPECT_NEAR(m.m[3].w, c, 1e-5);
}

TEST(LightScattering, MullerReflection)
{
    const otk::Transform4 m = calculateMullerReflection(std::cos(rad(45.0f)), std::cos(rad(28.125f)), 1.0f, 1.33f);
    const float pp = 0.031214129f, pm = -0.030182571f, p = -0.0079582815f;
    EXPECT_NEAR(m.m[0].x, pp, 1e-6);
    EXPECT_NEAR(m.m[0].y, pm, 1e-6);
    EXPECT_NEAR(m.m[1].x, pm, 1e-6);
    EXPECT_NEAR(m.m[1].y, pp, 1e-6);
    EXPECT_NEAR(m.m[2].z, p, 1e-6);
    EXPECT_NEAR(m.m[3].w, p, 1e-6);
}

TEST(LightScattering, MullerTransmission)
{
    // no interface: the identity
    expectNear(calculateMullerTransmission(1.0f, 1.0f, 1.0f, 1.0f), otk::identity(), 1e-6);
}

TEST(LightScattering, FresnelInteraction)
{
    const float3 normal = make_float3(0, 1, 0);
    const float3 incident = otk::normalize(make_float3(1, -1, 0));
    const float nAir = 1.0f, nIce = 1.33f;

    float3 axis;
    scatteringPlaneNormalAxis(incident, normal, axis);

    float cosTheta, cosThetaPrime;
    bool tir;
    calculateSnellsLaw(incident, normal, nAir, nIce, cosTheta, cosThetaPrime, tir);
    ASSERT_FALSE(tir);

    const float4 stokes = make_float4(1, 1, 0, 0);

    // A random sample close to 1 selects the reflected branch
    float4 s = stokes;
    float3 q = axis, d = incident;
    EXPECT_TRUE(calculateFresnelInteraction(s, q, d, normal, nAir, nIce, 0.999f));
    expectNear(d, calculateReflectedDirection(incident, normal), 1e-5);
    float4 expected = calculateMullerReflection(cosTheta, cosThetaPrime, nAir, nIce) * stokes;
    expectNear(s, expected / expected.x, 1e-5);

    // A random sample of 0 selects the transmitted branch
    s = stokes;
    q = axis;
    d = incident;
    EXPECT_FALSE(calculateFresnelInteraction(s, q, d, normal, nAir, nIce, 0.0f));
    expectNear(d, calculateRefractedDirection(incident, normal, cosTheta, cosThetaPrime, nAir, nIce), 1e-5);
    expected = calculateMullerTransmission(cosTheta, cosThetaPrime, nAir, nIce) * stokes;
    expectNear(s, expected / expected.x, 1e-5);
}

TEST(MediumHistory, AppendOverflowKeepsState)
{
    uint32_t packed = 0, size = 0;
    for (uint32_t medium = 1; medium <= 8; ++medium)
    {
        EXPECT_TRUE(appendMediumPacked(medium, size, packed));
    }
    EXPECT_FALSE(appendMediumPacked(9u, size, packed));
    EXPECT_EQ(size, 8u);
    for (uint32_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(getMediumFromPacked(packed, i), i + 1);
    }
    EXPECT_EQ(getMediumFromPacked(packed, 8), 0u);
}

TEST(MediumHistory, RemoveLastOccurrence)
{
    uint32_t packed = 0, size = 0;
    for (uint32_t medium : std::array<uint32_t, 5>{1, 2, 15, 2, 4})
    {
        ASSERT_TRUE(appendMediumPacked(medium, size, packed));
    }

    EXPECT_TRUE(removeLastOccurencePacked(2u, size, packed));
    uint32_t history[8] = {};
    unpackMediumHistory(packed, history);
    EXPECT_EQ(size, 4u);
    EXPECT_EQ(history[0], 1u);
    EXPECT_EQ(history[1], 2u);
    EXPECT_EQ(history[2], 15u);
    EXPECT_EQ(history[3], 4u);

    EXPECT_FALSE(removeLastOccurencePacked(7u, size, packed));
    EXPECT_EQ(size, 4u);

    for (uint32_t medium : {4u, 15u, 2u, 1u})
    {
        EXPECT_TRUE(removeLastOccurencePacked(medium, size, packed));
    }
    EXPECT_EQ(size, 0u);
    EXPECT_EQ(packed, 0u);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
