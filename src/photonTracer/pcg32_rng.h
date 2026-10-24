// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <cstdint>

#include "portable_math.h"

// PCG32 (XSH-RR 64/32) by M. E. O'Neill, https://www.pcg-random.org. Small state (16 bytes),
// passes BigCrush, and runs unchanged on the host and in CUDA code.
struct Pcg32State
{
    uint64_t state;
    uint64_t inc; // odd; selects the stream

    /// Same seeding as pcg32_srandom_r(): `initState` sets the start, `streamId` the stream
    PT_HD void seed(uint64_t initState, uint64_t streamId)
    {
        state = 0u;
        inc = (streamId << 1u) | 1u;
        nextUint32();
        state += initState;
        nextUint32();
    }

    PT_HD uint32_t nextUint32()
    {
        const uint64_t oldState = state;
        state = oldState * 6364136223846793005ull + inc;
        const uint32_t xorShifted = static_cast<uint32_t>(((oldState >> 18u) ^ oldState) >> 27u);
        const uint32_t rotation = static_cast<uint32_t>(oldState >> 59u);
        return (xorShifted >> rotation) | (xorShifted << ((~rotation + 1u) & 31u));
    }

    /// Uniform number in (0, 1] with 24 bits; never 0, so logf() of it is finite
    PT_HD float nextFloat()
    {
        return static_cast<float>((nextUint32() >> 8) + 1u) * (1.0f / 16777216.0f);
    }
};

/// Generator of one ray: a stream per (seed, sequence) and a start mixed from both
PT_HD inline Pcg32State makePcg32(uint32_t seed, uint32_t sequence)
{
    // SplitMix64 finalizer spreads neighbouring sequence numbers over the whole state
    uint64_t z = (static_cast<uint64_t>(seed) << 32 | sequence) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);

    Pcg32State rng;
    rng.seed(z, static_cast<uint64_t>(seed) << 32 | sequence);
    return rng;
}
