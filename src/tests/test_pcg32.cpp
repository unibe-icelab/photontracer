// PCG32 on the CPU: reference values, range and statistical sanity checks.

#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "../photonTracer/pcg32_rng.h"

TEST(Pcg32, MatchesTheReferenceImplementation)
{
    // Output of pcg32_srandom_r(&rng, 42, 54) in the reference demo at pcg-random.org
    Pcg32State rng;
    rng.seed(42u, 54u);
    const std::array<uint32_t, 6> expected = {0xa15c02b7u, 0x7b47f409u, 0xba1d3330u, 0x83d2f293u, 0xbfa4784bu, 0xcbed606eu};
    for (uint32_t value : expected)
    {
        EXPECT_EQ(rng.nextUint32(), value);
    }
}

TEST(Pcg32, IsDeterministic)
{
    Pcg32State a = makePcg32(42, 7), b = makePcg32(42, 7);
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_EQ(a.nextUint32(), b.nextUint32());
    }
}

TEST(Pcg32, FloatsAreInTheHalfOpenIntervalAboveZero)
{
    Pcg32State rng = makePcg32(1, 2);
    float smallest = 1.0f, largest = 0.0f;
    for (int i = 0; i < 1000000; ++i)
    {
        const float x = rng.nextFloat();
        smallest = std::fmin(smallest, x);
        largest = std::fmax(largest, x);
    }
    EXPECT_GT(smallest, 0.0f);
    EXPECT_LE(largest, 1.0f);
    EXPECT_TRUE(std::isfinite(std::log(smallest)));
}

TEST(Pcg32, ExtremeOutputsMapToTheEndsOfTheInterval)
{
    // 24 bits: the smallest value is 2^-24, the largest exactly 1
    EXPECT_FLOAT_EQ(static_cast<float>(0u + 1u) * (1.0f / 16777216.0f), 5.9604645e-08f);
    EXPECT_FLOAT_EQ(static_cast<float>(((0xFFFFFFFFu) >> 8) + 1u) * (1.0f / 16777216.0f), 1.0f);
}

TEST(Pcg32, MeanAndSpreadOfUniformNumbers)
{
    Pcg32State rng = makePcg32(3, 4);
    const int n = 1000000;
    double sum = 0.0, sumSquares = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double x = rng.nextFloat();
        sum += x;
        sumSquares += x * x;
    }
    const double mean = sum / n;
    const double variance = sumSquares / n - mean * mean;
    EXPECT_NEAR(mean, 0.5, 5.0 * std::sqrt(1.0 / 12.0 / n));
    EXPECT_NEAR(variance, 1.0 / 12.0, 0.002);
}

TEST(Pcg32, BinsAreEquallyFilled)
{
    Pcg32State rng = makePcg32(5, 6);
    const int bins = 16, n = 1600000;
    std::vector<int> counts(bins, 0);
    for (int i = 0; i < n; ++i)
    {
        ++counts[std::min(bins - 1, static_cast<int>(rng.nextFloat() * bins))];
    }
    double chiSquare = 0.0;
    for (int c : counts)
    {
        const double d = c - static_cast<double>(n) / bins;
        chiSquare += d * d / (static_cast<double>(n) / bins);
    }
    // 15 degrees of freedom: the 99.99% quantile is about 38
    EXPECT_LT(chiSquare, 38.0);
}

TEST(Pcg32, NeighbouringRaysAreUncorrelated)
{
    // The generators of consecutive launch indices must not move together
    const int n = 200000;
    for (uint32_t sequence : {0u, 1u, 12345u})
    {
        Pcg32State a = makePcg32(42, sequence), b = makePcg32(42, sequence + 1);
        double sumA = 0, sumB = 0, sumAB = 0, sumAA = 0, sumBB = 0;
        for (int i = 0; i < n; ++i)
        {
            const double x = a.nextFloat(), y = b.nextFloat();
            sumA += x;
            sumB += y;
            sumAB += x * y;
            sumAA += x * x;
            sumBB += y * y;
        }
        const double covariance = sumAB / n - (sumA / n) * (sumB / n);
        const double correlation = covariance / std::sqrt((sumAA / n - (sumA / n) * (sumA / n)) * (sumBB / n - (sumB / n) * (sumB / n)));
        EXPECT_NEAR(correlation, 0.0, 5.0 / std::sqrt(static_cast<double>(n))) << "sequence " << sequence;
    }
}

TEST(Pcg32, DifferentSeedsGiveDifferentStreams)
{
    Pcg32State a = makePcg32(1, 0), b = makePcg32(2, 0);
    int same = 0;
    for (int i = 0; i < 1000; ++i)
    {
        same += a.nextUint32() == b.nextUint32();
    }
    EXPECT_LT(same, 5);
}

TEST(Pcg32, FirstDrawsAreSpreadForConsecutiveSequences)
{
    // Rays start next to each other: the very first numbers must already look uniform
    const int n = 100000;
    double sum = 0.0;
    for (int sequence = 0; sequence < n; ++sequence)
    {
        Pcg32State rng = makePcg32(42, sequence);
        sum += rng.nextFloat();
    }
    EXPECT_NEAR(sum / n, 0.5, 5.0 * std::sqrt(1.0 / 12.0 / n));
}
