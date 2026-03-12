// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>
#include <curand_kernel.h>
#include <cstdint>
#include <array>

#include "light_scattering.cuh"

#pragma once

struct RayState
{
    uint32_t numberOfWarnings;         // 21 bit (0-2097151)
    uint32_t currentMedium;            // 4 bit (0-15)
    uint32_t currentMediumHistorySize; // 4 bit (0-15)
    uint32_t absorbed;                 // 2 bit (0-15)
    bool done;                         // 1 bit
};

struct RayData
{
    float3 direction;
    float3 origin;
    float4 stokesVector;
    float3 qMinusAxis;
    double opticalPathLength;

    curandStateMRG32k3a randState;
    RayState state;
    uint32_t mediumHistory[8]; // 8 layers of history, each 4 bits
};

struct DensityData
{
    float3 direction;
    float3 origin;
    uint32_t numberOfFrontFaces;
    uint32_t numberOfBackFaces;
    bool done;
};

static __forceinline__ __device__ void *unpackPointer(uint32_t i0, uint32_t i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void *ptr = reinterpret_cast<void *>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packPointer(void *ptr, uint32_t &i0, uint32_t &i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void packFloat4(const float4 &vector, uint32_t *vectorPacked)
{
    vectorPacked[0] = __float_as_uint(vector.x);
    vectorPacked[1] = __float_as_uint(vector.y);
    vectorPacked[2] = __float_as_uint(vector.z);
    vectorPacked[3] = __float_as_uint(vector.w);
}

static __forceinline__ __device__ void packFloat3(const float3 &vector, uint32_t *vectorPacked)
{
    vectorPacked[0] = __float_as_uint(vector.x);
    vectorPacked[1] = __float_as_uint(vector.y);
    vectorPacked[2] = __float_as_uint(vector.z);
}

static __forceinline__ __device__ float4 unpackFloat4(const uint32_t *vectorPacked)
{
    return make_float4(
        __uint_as_float(vectorPacked[0]),
        __uint_as_float(vectorPacked[1]),
        __uint_as_float(vectorPacked[2]),
        __uint_as_float(vectorPacked[3]));
}

static __forceinline__ __device__ float3 unpackFloat3(const uint32_t *vectorPacked)
{
    return make_float3(
        __uint_as_float(vectorPacked[0]),
        __uint_as_float(vectorPacked[1]),
        __uint_as_float(vectorPacked[2]));
}

static __forceinline__ __device__ uint32_t packRayState(const RayState &state)
{
    uint32_t packedState = 0;
    packedState |= (state.numberOfWarnings & 0x1FFFFF) << 11;   // 21 bits for numberOfWarnings
    packedState |= (state.currentMedium & 0xF) << 7;            // 4 bits for currentMedium
    packedState |= (state.currentMediumHistorySize & 0xF) << 3; // 4 bits for currentMediumHistorySize
    packedState |= (state.absorbed & 0x3) << 1;                 // 2 bits for absorbed
    packedState |= (state.done ? 1 : 0);                        // 1 bit for done
    return packedState;
}

static __forceinline__ __device__ RayState unpackRayState(uint32_t packedState)
{
    RayState state;
    state.numberOfWarnings = (packedState >> 11) & 0x1FFFFF;   // Extract 21 bits for numberOfWarnings
    state.currentMedium = (packedState >> 7) & 0xF;            // Extract 4 bits for currentMedium
    state.currentMediumHistorySize = (packedState >> 3) & 0xF; // Extract 4 bits for currentMediumHistorySize
    state.absorbed = (packedState >> 1) & 0x3;                 // Extract 2 bits for absorbed
    state.done = (packedState & 0x1) != 0;                     // Extract the last bit for done
    return state;
}

// Function to save the state, boxmuller flag and boxmuller extra not saved
static __forceinline__ __device__ void packCuRandStateMRG32k3a(const curandStateMRG32k3a *state, uint32_t *statePacked)
{
    // Copy each of the 6 internal state variables into storage
    statePacked[0] = state->s1[0];
    statePacked[1] = state->s1[1];
    statePacked[2] = state->s1[2];
    statePacked[3] = state->s2[0];
    statePacked[4] = state->s2[1];
    statePacked[5] = state->s2[2];
}

// Function to restore the state of curandStateMRG32k3a from 6 uint32_ts, boxmuller flag and boxmuller extra not restored
static __forceinline__ __device__ curandStateMRG32k3a unpackCuRandStateMRG32k3a(uint32_t *statePacked)
{
    curandStateMRG32k3a state;
    state.s1[0] = statePacked[0];
    state.s1[1] = statePacked[1];
    state.s1[2] = statePacked[2];
    state.s2[0] = statePacked[3];
    state.s2[1] = statePacked[4];
    state.s2[2] = statePacked[5];

    state.boxmuller_flag = 0;
    state.boxmuller_extra = 0.0f;
    state.boxmuller_flag_double = 0;
    state.boxmuller_extra_double = 0.0;

    return state;
}

// function to save the state of curandStateXORWOW, boxmuller flag and boxmuller extra not saved
static __forceinline__ __device__ void packCuRandStateXORWOW(const curandStateXORWOW *state, uint32_t *statePacked)
{
    // Copy each of the 6 internal state variables into storage
    statePacked[0] = state->d;
    statePacked[1] = state->v[0];
    statePacked[2] = state->v[1];
    statePacked[3] = state->v[2];
    statePacked[4] = state->v[3];
    statePacked[5] = state->v[4];
}

// Function to restore the state of curandStateXORWOW from 6 uint32_ts, boxmuller flag and boxmuller extra not restored
static __forceinline__ __device__ curandStateXORWOW unpackCuRandStateXORWOW(uint32_t *statePacked)
{
    curandStateXORWOW state;
    state.d = statePacked[0];
    state.v[0] = statePacked[1];
    state.v[1] = statePacked[2];
    state.v[2] = statePacked[3];
    state.v[3] = statePacked[4];
    state.v[4] = statePacked[5];

    state.boxmuller_flag = 0;
    state.boxmuller_extra = 0.0f;
    state.boxmuller_flag_double = 0;
    state.boxmuller_extra_double = 0.0;

    return state;
}

static __forceinline__ __device__ float3 getRayOrigin()
{
    return make_float3(
        __uint_as_float(optixGetPayload_0()),
        __uint_as_float(optixGetPayload_1()),
        __uint_as_float(optixGetPayload_2()));
}

static __forceinline__ __device__ void setRayOrigin(const float3 &origin)
{
    optixSetPayload_0(__float_as_uint(origin.x));
    optixSetPayload_1(__float_as_uint(origin.y));
    optixSetPayload_2(__float_as_uint(origin.z));
}

static __forceinline__ __device__ float3 getRayDirection()
{
    return make_float3(
        __uint_as_float(optixGetPayload_3()),
        __uint_as_float(optixGetPayload_4()),
        __uint_as_float(optixGetPayload_5()));
}

static __forceinline__ __device__ void setRayDirection(const float3 &direction)
{
    optixSetPayload_3(__float_as_uint(direction.x));
    optixSetPayload_4(__float_as_uint(direction.y));
    optixSetPayload_5(__float_as_uint(direction.z));
}

static __forceinline__ __device__ float4 getStokesVector()
{
    return make_float4(
        __uint_as_float(optixGetPayload_6()),
        __uint_as_float(optixGetPayload_7()),
        __uint_as_float(optixGetPayload_8()),
        __uint_as_float(optixGetPayload_9()));
}

static __forceinline__ __device__ void setStokesVector(const float4 &stokesVector)
{
    optixSetPayload_6(__float_as_uint(stokesVector.x));
    optixSetPayload_7(__float_as_uint(stokesVector.y));
    optixSetPayload_8(__float_as_uint(stokesVector.z));
    optixSetPayload_9(__float_as_uint(stokesVector.w));
}

static __forceinline__ __device__ float3 getQMinusAxis()
{
    return make_float3(
        __uint_as_float(optixGetPayload_10()),
        __uint_as_float(optixGetPayload_11()),
        __uint_as_float(optixGetPayload_12()));
}

static __forceinline__ __device__ void setQDirection(const float3 &dir)
{
    optixSetPayload_10(__float_as_uint(dir.x));
    optixSetPayload_11(__float_as_uint(dir.y));
    optixSetPayload_12(__float_as_uint(dir.z));
}

static __forceinline__ __device__ void setRayState(RayState state)
{
    optixSetPayload_13(packRayState(state));
}

static __forceinline__ __device__ RayState getRayState()
{
    return unpackRayState(optixGetPayload_13());
}

static __forceinline__ __device__ void setDone(bool done)
{
    uint32_t packedState = optixGetPayload_13();
    packedState &= ~(1U);            // Clear the last bit
    packedState |= (done ? 1U : 0U); // Set the last bit to done
    optixSetPayload_13(packedState);
}

static __forceinline__ __device__ void setLayerHistory(uint32_t mediumHistory[8])
{
    optixSetPayload_14(packMediumHistory(mediumHistory));
}

static __forceinline__ __device__ void getMediumHistory(uint32_t mediumHistory[8])
{
    unpackMediumHistory(optixGetPayload_14(), mediumHistory);
}

static __forceinline__ __device__ uint32_t getMedium(uint32_t i)
{
    return getMediumFromPacked(optixGetPayload_14(), i);
}

static __forceinline__ __device__ bool appendMedium(uint32_t medium, uint32_t &mediumHistorySize)
{
    uint32_t packedMediumHistory = optixGetPayload_14();
    if (!appendMediumPacked(medium, mediumHistorySize, packedMediumHistory))
    {
        return false;
    }
    optixSetPayload_14(packedMediumHistory);
    return true;
}

static __forceinline__ __device__ bool removeLastOccurence(uint32_t medium, uint32_t &mediumHistorySize)
{
    uint32_t packedMediumHistory = optixGetPayload_14();
    const bool found = removeLastOccurencePacked(medium, mediumHistorySize, packedMediumHistory);
    optixSetPayload_14(packedMediumHistory);
    return found;
}

static __forceinline__ __device__ curandStateMRG32k3a getCuRandStateMRG32k3a()
{
    uint32_t statePacked[6];
    statePacked[0] = optixGetPayload_15();
    statePacked[1] = optixGetPayload_16();
    statePacked[2] = optixGetPayload_17();
    statePacked[3] = optixGetPayload_18();
    statePacked[4] = optixGetPayload_19();
    statePacked[5] = optixGetPayload_20();
    return unpackCuRandStateMRG32k3a(statePacked);
}

static __forceinline__ __device__ void setCuRandStateMRG32k3a(const curandStateMRG32k3a &randState)
{
    uint32_t statePacked[6];
    packCuRandStateMRG32k3a(&randState, statePacked);
    optixSetPayload_15(statePacked[0]);
    optixSetPayload_16(statePacked[1]);
    optixSetPayload_17(statePacked[2]);
    optixSetPayload_18(statePacked[3]);
    optixSetPayload_19(statePacked[4]);
    optixSetPayload_20(statePacked[5]);
}

static __forceinline__ __device__ void setCuRandStateXORWOW(const curandStateXORWOW &randState)
{
    uint32_t statePacked[6];
    packCuRandStateXORWOW(&randState, statePacked);
    optixSetPayload_15(statePacked[0]);
    optixSetPayload_16(statePacked[1]);
    optixSetPayload_17(statePacked[2]);
    optixSetPayload_18(statePacked[3]);
    optixSetPayload_19(statePacked[4]);
    optixSetPayload_20(statePacked[5]);
}

static __forceinline__ __device__ curandStateXORWOW getCuRandStateXORWOW()
{
    uint32_t statePacked[6];
    statePacked[0] = optixGetPayload_15();
    statePacked[1] = optixGetPayload_16();
    statePacked[2] = optixGetPayload_17();
    statePacked[3] = optixGetPayload_18();
    statePacked[4] = optixGetPayload_19();
    statePacked[5] = optixGetPayload_20();
    return unpackCuRandStateXORWOW(statePacked);
}

static __forceinline__ __device__ float getOpticalPathLength()
{
    return __uint_as_float(optixGetPayload_21());
}

static __forceinline__ __device__ void setOpticalPathLength(const float &opl)
{
    optixSetPayload_21(__float_as_uint(opl));
}
