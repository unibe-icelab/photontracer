// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <cstdint>

#include "payload_layout.h"

// The random number generator of the OptiX programs: its state, how a ray's generator is
// seeded and drawn from, and how the state travels in the payload between programs.
// PHOTONTRACER_RNG_PCG32 is the default; PHOTONTRACER_RNG_MRG32K3A is curand's MRG32k3a,
// the generator of the 1.0 releases.

#if defined(PHOTONTRACER_RNG_PCG32)

#include "pcg32_rng.h"

using RandState = Pcg32State;

/// Number of 32-bit payload words that hold a RandState
constexpr int RAND_STATE_WORDS = PCG32_PAYLOAD_WORDS;

/// State of the generator of one ray; `sequence` is the linear launch index
__device__ __forceinline__ RandState randInit(unsigned int seed, unsigned int sequence)
{
    return makePcg32(seed, sequence);
}

/// Uniform number in (0, 1]
__device__ __forceinline__ float randNext(RandState &state)
{
    return state.nextFloat();
}

__device__ __forceinline__ void packRandState(const RandState &state, uint32_t *words)
{
    words[0] = static_cast<uint32_t>(state.state);
    words[1] = static_cast<uint32_t>(state.state >> 32);
    words[2] = static_cast<uint32_t>(state.inc);
    words[3] = static_cast<uint32_t>(state.inc >> 32);
}

__device__ __forceinline__ RandState unpackRandState(const uint32_t *words)
{
    RandState state;
    state.state = static_cast<uint64_t>(words[1]) << 32 | words[0];
    state.inc = static_cast<uint64_t>(words[3]) << 32 | words[2];
    return state;
}

#elif defined(PHOTONTRACER_RNG_MRG32K3A)

#include <curand_kernel.h>

using RandState = curandStateMRG32k3a;

/// Number of 32-bit payload words that hold a RandState
constexpr int RAND_STATE_WORDS = MRG32K3A_PAYLOAD_WORDS;

/// State of the generator of one ray; `sequence` is the linear launch index
__device__ __forceinline__ RandState randInit(unsigned int seed, unsigned int sequence)
{
    RandState state;
    curand_init(seed, sequence, 0, &state);
    return state;
}

/// Uniform number in (0, 1]
__device__ __forceinline__ float randNext(RandState &state)
{
    return curand_uniform(&state);
}

__device__ __forceinline__ void packRandState(const RandState &state, uint32_t *words)
{
    words[0] = state.s1[0];
    words[1] = state.s1[1];
    words[2] = state.s1[2];
    words[3] = state.s2[0];
    words[4] = state.s2[1];
    words[5] = state.s2[2];
}

// The Box-Muller cache is not carried along; the physics only draws uniform numbers
__device__ __forceinline__ RandState unpackRandState(const uint32_t *words)
{
    RandState state;
    state.s1[0] = words[0];
    state.s1[1] = words[1];
    state.s1[2] = words[2];
    state.s2[0] = words[3];
    state.s2[1] = words[4];
    state.s2[2] = words[5];

    state.boxmuller_flag = 0;
    state.boxmuller_extra = 0.0f;
    state.boxmuller_flag_double = 0;
    state.boxmuller_extra_double = 0.0;

    return state;
}

#else
#error "Define PHOTONTRACER_RNG_PCG32 or PHOTONTRACER_RNG_MRG32K3A"
#endif
