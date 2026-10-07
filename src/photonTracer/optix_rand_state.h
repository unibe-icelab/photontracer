// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <curand_kernel.h>
#include <cstdint>

// The random number generator of the OptiX programs: its state, how a ray's generator is
// seeded and drawn from, and how the state travels in the payload between programs.

using RandState = curandStateMRG32k3a;

/// Number of 32-bit payload words that hold a RandState
constexpr int RAND_STATE_WORDS = 6;

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
