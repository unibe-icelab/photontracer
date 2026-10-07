// Ray sources, source polarization and HEALPix binning on the CPU, with scripted random numbers.

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>

#include "../photonTracer/light_trace.h"
#include "../photonTracer/ray_generation.h"

namespace
{
struct ScriptedSample
{
    std::vector<float> values;
    size_t used = 0;

    ScriptedSample(std::initializer_list<float> v) : values(v) {}
    float operator()() { return values.at(used++); }
};

void expectNear(float3 a, float3 b, float tol = 1e-5f)
{
    EXPECT_NEAR(a.x, b.x, tol);
    EXPECT_NEAR(a.y, b.y, tol);
    EXPECT_NEAR(a.z, b.z, tol);
}

RayGeneratorData::CameraSource makeCamera(unsigned int width, unsigned int height, unsigned int samples, bool defocus)
{
    RayGeneratorData::CameraSource camera{};
    camera.center = make_float3(0, 0, 10);
    camera.pixel00 = make_float3(-1.5f, 1.5f, 9.0f); // 4 x 4 pixels on the plane z = 9
    camera.pixelDeltaU = make_float3(1, 0, 0);
    camera.pixelDeltaV = make_float3(0, -1, 0);
    camera.defocusDiskU = make_float3(0.1f, 0, 0);
    camera.defocusDiskV = make_float3(0, 0.1f, 0);
    camera.imageWidth = width;
    camera.imageHeight = height;
    camera.samplesPerPixel = samples;
    camera.enableDefocus = defocus;
    return camera;
}
} // namespace

TEST(Healpix, PolesAndRange)
{
    for (int nside : {1, 2, 4, 16})
    {
        const int npix = 12 * nside * nside;
        const int north = healpixAng2PixRing(nside, make_float3(0, 0, 1));
        const int south = healpixAng2PixRing(nside, make_float3(0, 0, -1));
        EXPECT_GE(north, 0);
        EXPECT_LT(north, 4);
        EXPECT_GE(south, npix - 4);
        EXPECT_LT(south, npix);
    }
}

TEST(Healpix, PixelsHaveEqualArea)
{
    // A Fibonacci sphere is uniform, so every pixel should receive about the same number of points
    const int nside = 4;
    const int npix = 12 * nside * nside;
    const int points = 200000;
    std::vector<int> counts(npix, 0);
    const float golden = 3.14159265f * (3.0f - std::sqrt(5.0f));
    for (int i = 0; i < points; ++i)
    {
        const float z = 1.0f - 2.0f * (i + 0.5f) / points;
        const float r = std::sqrt(1.0f - z * z);
        const float phi = golden * i;
        const int pix = healpixAng2PixRing(nside, make_float3(r * std::cos(phi), r * std::sin(phi), z));
        ASSERT_GE(pix, 0);
        ASSERT_LT(pix, npix);
        ++counts[pix];
    }
    const double expected = static_cast<double>(points) / npix;
    for (int c : counts)
    {
        EXPECT_NEAR(c, expected, 0.1 * expected);
    }
}

TEST(Healpix, NeighbouringLongitudesGiveNeighbouringPixelsOnTheEquator)
{
    const int nside = 4;
    int previous = healpixAng2PixRing(nside, make_float3(1, 0.001f, 0));
    for (int i = 1; i < 4 * nside; ++i)
    {
        const float phi = 2.0f * 3.14159265f * (i + 0.5f) / (4 * nside);
        const int pix = healpixAng2PixRing(nside, make_float3(std::cos(phi), std::sin(phi), 0));
        EXPECT_EQ(pix, previous + 1) << "i = " << i;
        previous = pix;
    }
}

TEST(RayGeneration, ParallelCenterAndEdge)
{
    RayGeneratorData::ParallelSource source{};
    source.origin = make_float3(1, 2, 3);
    source.direction = make_float3(0, 0, -1);
    source.offsetRadius = 2.0f;

    float3 origin, direction;
    ScriptedSample center{0.5f, 0.5f};
    computeRayParallel(source, center, origin, direction);
    expectNear(origin, source.origin);
    expectNear(direction, source.direction);
    EXPECT_EQ(center.used, 2u);

    // x = 1, y = 0 lies on the rim
    ScriptedSample edge{1.0f, 0.5f};
    computeRayParallel(source, edge, origin, direction);
    const float3 offset = origin - source.origin;
    EXPECT_NEAR(otk::length(offset), 2.0f, 1e-5);
    EXPECT_NEAR(otk::dot(offset, source.direction), 0.0f, 1e-5);
}

TEST(RayGeneration, ParallelRejectsPointsOutsideTheDisk)
{
    RayGeneratorData::ParallelSource source{};
    source.direction = make_float3(0, 0, -1);
    source.offsetRadius = 1.0f;

    float3 origin, direction;
    // (1, 1) is outside the unit disk and is redrawn
    ScriptedSample sample{1.0f, 1.0f, 0.5f, 0.5f};
    computeRayParallel(source, sample, origin, direction);
    EXPECT_EQ(sample.used, 4u);
    expectNear(origin, make_float3(0, 0, 0));
}

TEST(RayGeneration, ParallelWorksForABeamAlongY)
{
    RayGeneratorData::ParallelSource source{};
    source.direction = make_float3(0, 1, 0);
    source.offsetRadius = 1.0f;

    float3 origin, direction;
    ScriptedSample sample{0.9f, 0.6f};
    computeRayParallel(source, sample, origin, direction);
    EXPECT_NEAR(otk::dot(origin, source.direction), 0.0f, 1e-5);
    EXPECT_GT(otk::length(origin), 0.0f);
}

TEST(RayGeneration, IsotropicStartsOnTheSphereAndLooksInward)
{
    RayGeneratorData::IsotropicSource source{};
    source.center = make_float3(0, 0, 0);
    source.sourceRadius = 30.0f;
    source.offsetRadius = 5.0f;

    float3 origin, direction;
    // direction sample (0.75, 0.75, 0.75) is the point (0.5, 0.5, 0.5); offset sample (0.5, 0.5) is zero
    ScriptedSample sample{0.75f, 0.75f, 0.75f, 0.5f, 0.5f};
    computeRayIsotropic(source, sample, origin, direction);

    const float3 outward = otk::normalize(make_float3(1, 1, 1));
    expectNear(origin, 30.0f * outward);
    expectNear(direction, -outward);
    EXPECT_EQ(sample.used, 5u);
}

TEST(RayGeneration, IsotropicOffsetStaysWithinTheDisk)
{
    RayGeneratorData::IsotropicSource source{};
    source.sourceRadius = 30.0f;
    source.offsetRadius = 5.0f;

    float3 origin, direction;
    ScriptedSample sample{0.75f, 0.75f, 0.75f, 0.9f, 0.6f};
    computeRayIsotropic(source, sample, origin, direction);

    const float3 onSphere = 30.0f * (-direction);
    const float3 offset = origin - onSphere;
    EXPECT_LE(otk::length(offset), 5.0f + 1e-4f);
    EXPECT_NEAR(otk::dot(offset, direction), 0.0f, 1e-4);
}

TEST(RayGeneration, CameraRayIndexOrdersSamplesWithinPixels)
{
    const auto camera = makeCamera(4, 4, 2, false);
    EXPECT_EQ(cameraRayIndex(camera, make_uint3(0, 0, 0)), 0u);
    EXPECT_EQ(cameraRayIndex(camera, make_uint3(1, 0, 0)), 1u);
    EXPECT_EQ(cameraRayIndex(camera, make_uint3(1, 3, 2)), (2u * 4 + 3) * 2 + 1);
    // indices past the image are clamped to the last sample and pixel
    EXPECT_EQ(cameraRayIndex(camera, make_uint3(9, 9, 9)), (3u * 4 + 3) * 2 + 1);
}

TEST(RayGeneration, CameraRayThroughPixelCenter)
{
    const auto camera = makeCamera(4, 4, 1, false);
    float3 origin, direction;

    // pixel (1, 2) with no jitter: the plane point is (-0.5, -0.5, 9)
    ScriptedSample sample{0.5f, 0.5f};
    computeRayCamera(camera, sample, cameraRayIndex(camera, make_uint3(0, 1, 2)), origin, direction);
    expectNear(origin, camera.center);
    expectNear(direction, otk::normalize(make_float3(-0.5f, -0.5f, -1.0f)));
    EXPECT_EQ(sample.used, 2u);
}

TEST(RayGeneration, CameraDefocusMovesTheOriginAndUsesTwoMoreSamples)
{
    const auto camera = makeCamera(4, 4, 1, true);
    float3 origin, direction;
    ScriptedSample sample{0.5f, 0.5f, 0.25f, 0.81f};
    computeRayCamera(camera, sample, 0, origin, direction);

    EXPECT_EQ(sample.used, 4u);
    const float angle = 2.0f * 3.14159265f * 0.25f;
    const float radius = std::sqrt(0.81f);
    expectNear(origin, camera.center + make_float3(std::cos(angle) * radius * 0.1f, std::sin(angle) * radius * 0.1f, 0.0f), 1e-5f);
}

TEST(RayGeneration, ComputeRayDispatchesAndReportsUnknownTypes)
{
    RayGeneratorData data{};
    data.camera = makeCamera(4, 4, 2, false);
    float3 origin, direction;
    uint32_t index = 0;
    ScriptedSample sample{0.5f, 0.5f};

    EXPECT_TRUE(computeRay(RAYGEN_CAMERA, data, sample, make_uint3(1, 3, 2), index, origin, direction));
    EXPECT_EQ(index, (2u * 4 + 3) * 2 + 1); // the camera replaces the linear index

    index = 7;
    EXPECT_FALSE(computeRay(static_cast<RayGeneratorType>(99), data, sample, make_uint3(0, 0, 0), index, origin, direction));
    EXPECT_EQ(index, 7u);
}

TEST(SourcePolarization, FixedStokesVectorUsesNoRandomNumbers)
{
    ScriptedSample sample{};
    const float4 stokes = randomizeStokesVector(make_float4(1, 0.2f, 0.3f, 0.4f), sample);
    EXPECT_EQ(sample.used, 0u);
    EXPECT_FLOAT_EQ(stokes.y, 0.2f);
    EXPECT_FLOAT_EQ(stokes.w, 0.4f);
}

TEST(SourcePolarization, RandomLinearPolarizationIsFullyLinear)
{
    ScriptedSample sample{0.3f};
    const float4 stokes = randomizeStokesVector(make_float4(1, NAN, NAN, 0), sample);
    EXPECT_EQ(sample.used, 1u);
    EXPECT_NEAR(stokes.y * stokes.y + stokes.z * stokes.z, 1.0f, 1e-5);
    EXPECT_FLOAT_EQ(stokes.w, 0.0f);
}

TEST(SourcePolarization, RandomCircularPolarizationIsBounded)
{
    ScriptedSample sample{0.1f};
    const float4 stokes = randomizeStokesVector(make_float4(1, 0, 0, NAN), sample);
    EXPECT_EQ(sample.used, 1u);
    EXPECT_NEAR(stokes.w, std::cos(2.0f * 3.14159265f * 0.1f), 1e-5);
}

TEST(SourcePolarization, InitialQMinusAxis)
{
    const float3 k = make_float3(0, 0, -1);

    // a usable seed is projected perpendicular to the ray
    expectNear(initialQMinusAxis(make_float3(0, 1, 1), k), make_float3(0, 1, 0));

    // no seed: e_x
    expectNear(initialQMinusAxis(make_float3(NAN, NAN, NAN), k), make_float3(1, 0, 0));

    // seed parallel to the ray: e_x
    expectNear(initialQMinusAxis(make_float3(0, 0, 1), k), make_float3(1, 0, 0));

    // a ray along x cannot use e_x: e_y
    expectNear(initialQMinusAxis(make_float3(NAN, NAN, NAN), make_float3(1, 0, 0)), make_float3(0, 1, 0));
}

TEST(DensitySampling, OriginIsInsideTheBox)
{
    ScriptedSample sample{0.0f, 0.5f, 1.0f};
    const float3 origin = computeDensitySampleOrigin(make_float3(-1, -2, -3), make_float3(1, 2, 3), sample);
    EXPECT_EQ(sample.used, 3u);
    expectNear(origin, make_float3(-1, 0, 3));
}

namespace
{
struct Buffers
{
    std::vector<float3> lastDirection, lastPosition, sourceDirection, sourcePosition, qMinusAxisIn;
    std::vector<float4> stokes, stokesIn;
    std::vector<int> rayState, lastMedium;
    std::vector<uint32_t> warnings, scatteringCount;
    std::vector<double> opl;
    std::vector<float> angle;

    Buffers() : lastDirection(1), lastPosition(1), sourceDirection(1), sourcePosition(1), qMinusAxisIn(1), stokes(1), stokesIn(1),
                rayState(1, -1), lastMedium(1, -1), warnings(1, 99), scatteringCount(1, 99), opl(1, -1.0), angle(1, -1.0f) {}

    DeviceOutputBuffers views()
    {
        DeviceOutputBuffers out{};
        out.lastDirection = lastDirection.data();
        out.lastPosition = lastPosition.data();
        out.sourceDirection = sourceDirection.data();
        out.sourcePosition = sourcePosition.data();
        out.qMinusAxisIn = qMinusAxisIn.data();
        out.stokesVector = stokes.data();
        out.stokesVectorIn = stokesIn.data();
        out.ray_state = rayState.data();
        out.lastMediumID = lastMedium.data();
        out.numberOfWarnings = warnings.data();
        out.scatteringCount = scatteringCount.data();
        out.opticalPathLength = opl.data();
        out.scatteringAngle = angle.data();
        return out;
    }
};
} // namespace

TEST(LightTrace, InitializeWritesOnlyTheRequestedSourceOutputs)
{
    Buffers buffers;
    ScriptedSample sample{};
    RayData prd{};
    float4 stokesIn;
    float3 qIn;

    initializeLightTraceRay(OUT_SOURCE_POSITION, buffers.views(), 0, make_float3(1, 2, 3), make_float3(0, 0, -1),
                            make_float4(1, 0, 0, 0), make_float3(NAN, NAN, NAN), sample, prd, stokesIn, qIn);

    expectNear(buffers.sourcePosition[0], make_float3(1, 2, 3));
    EXPECT_FLOAT_EQ(buffers.sourceDirection[0].z, 0.0f); // not requested, untouched
    EXPECT_FLOAT_EQ(buffers.qMinusAxisIn[0].x, 0.0f);

    expectNear(prd.origin, make_float3(1, 2, 3));
    expectNear(prd.direction, make_float3(0, 0, -1));
    expectNear(prd.qMinusAxis, make_float3(1, 0, 0));
    EXPECT_EQ(prd.state.currentMedium, 0u);
    EXPECT_EQ(prd.state.currentMediumHistorySize, 0u);
    EXPECT_EQ(prd.state.absorbed, 0u);
    EXPECT_FALSE(prd.state.done);
    EXPECT_EQ(prd.packedMediumHistory, 0u);
    EXPECT_EQ(prd.opticalPathLength, 0.0);
}

TEST(LightTrace, InitializeKeepsTheRandomOrder)
{
    // the source polarization is drawn after the ray itself
    Buffers buffers;
    ScriptedSample sample{0.3f};
    RayData prd{};
    float4 stokesIn;
    float3 qIn;
    initializeLightTraceRay(OUT_Q_MINUS_AXIS_IN, buffers.views(), 0, make_float3(0, 0, 5), make_float3(0, 0, -1),
                            make_float4(1, NAN, NAN, 0), make_float3(0, 1, 0), sample, prd, stokesIn, qIn);
    EXPECT_EQ(sample.used, 1u);
    expectNear(buffers.qMinusAxisIn[0], make_float3(0, 1, 0));
    EXPECT_NEAR(stokesIn.y * stokesIn.y + stokesIn.z * stokesIn.z, 1.0f, 1e-5);
    EXPECT_FLOAT_EQ(prd.stokesVector.y, stokesIn.y);
}

TEST(LightTrace, WriteOutputsByFlag)
{
    Buffers buffers;
    RayData prd{};
    prd.direction = make_float3(0, 0, 1); // turned around
    prd.origin = make_float3(4, 5, 6);
    prd.stokesVector = make_float4(1, 0.5f, 0, 0);
    prd.qMinusAxis = make_float3(1, 0, 0);
    prd.opticalPathLength = 12.5;
    prd.state.absorbed = 2;
    prd.state.currentMedium = 3;
    prd.state.numberOfWarnings = 4;

    const uint32_t flags = OUT_LAST_DIRECTION | OUT_LAST_POSITION | OUT_RAY_STATE | OUT_LAST_MEDIUM_ID | OUT_SCATTERING_COUNT |
                           OUT_NUMBER_OF_WARNINGS | OUT_OPTICAL_PATH_LENGTH | OUT_SCATTERING_ANGLE;
    writeMainTraceOutputs(flags, buffers.views(), 0, prd, make_float3(0, 0, -1), make_float4(1, 0, 0, 0), make_float3(1, 0, 0), 7);

    expectNear(buffers.lastDirection[0], make_float3(0, 0, 1));
    expectNear(buffers.lastPosition[0], make_float3(4, 5, 6));
    EXPECT_EQ(buffers.rayState[0], 2);
    EXPECT_EQ(buffers.lastMedium[0], 3);
    EXPECT_EQ(buffers.scatteringCount[0], 7u);
    EXPECT_EQ(buffers.warnings[0], 4u);
    EXPECT_DOUBLE_EQ(buffers.opl[0], 12.5);
    EXPECT_NEAR(buffers.angle[0], 3.14159265f, 1e-5); // straight back
    EXPECT_FLOAT_EQ(buffers.stokes[0].x, 0.0f);       // not requested, untouched
}

TEST(LightTrace, StokesOutputIsUnchangedWithoutAScatteringPlane)
{
    Buffers buffers;
    RayData prd{};
    prd.direction = make_float3(0, 0, -1); // not deflected, so there is no scattering plane
    prd.stokesVector = make_float4(1, 0.3f, 0.2f, 0.1f);
    prd.qMinusAxis = make_float3(1, 0, 0);

    writeMainTraceOutputs(OUT_STOKES_VECTOR | OUT_STOKES_VECTOR_IN, buffers.views(), 0, prd,
                          make_float3(0, 0, -1), make_float4(1, 0.6f, 0, 0), make_float3(1, 0, 0), 0);

    EXPECT_FLOAT_EQ(buffers.stokes[0].y, 0.3f);
    EXPECT_FLOAT_EQ(buffers.stokes[0].w, 0.1f);
    EXPECT_FLOAT_EQ(buffers.stokesIn[0].y, 0.6f);
}
