// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

// Vector types and math used by the physics headers. With CUDA these come from
// CUDA and the OptiX Toolkit; with PHOTONTRACER_NO_CUDA a small host-only
// replacement is provided so the same code builds without a CUDA toolkit.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>

#if !defined(PHOTONTRACER_NO_CUDA)

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/Transform4.h>

#define PT_HD __host__ __device__
#define PT_INLINE __forceinline__

#else

#define PT_HD
#define PT_INLINE inline

struct float3
{
    float x, y, z;
};

struct float4
{
    float x, y, z, w;
};

inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }

using std::isfinite;
using std::isnan;

inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }

inline float3 operator-(float3 a) { return {-a.x, -a.y, -a.z}; }
inline float3 operator+(float3 a, float3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline float3 operator-(float3 a, float3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline float3 operator*(float3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float3 operator*(float s, float3 a) { return a * s; }
inline float3 operator/(float3 a, float s) { return a * (1.0f / s); }
inline float3 &operator+=(float3 &a, float3 b) { return a = a + b; }
inline float3 &operator-=(float3 &a, float3 b) { return a = a - b; }
inline float3 &operator*=(float3 &a, float s) { return a = a * s; }

inline float4 operator-(float4 a) { return {-a.x, -a.y, -a.z, -a.w}; }
inline float4 operator+(float4 a, float4 b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
inline float4 operator-(float4 a, float4 b) { return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
inline float4 operator*(float4 a, float s) { return {a.x * s, a.y * s, a.z * s, a.w * s}; }
inline float4 operator*(float s, float4 a) { return a * s; }
inline float4 operator/(float4 a, float s) { return a * (1.0f / s); }

namespace otk
{
inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline float length(float3 v) { return std::sqrt(dot(v, v)); }
inline float3 normalize(float3 v) { return v * (1.0f / length(v)); }
inline float3 cross(float3 a, float3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Row-major 4x4 matrix, as in OptiXToolkit
class Transform4
{
public:
    float4 m[4];
};

inline Transform4 identity()
{
    return {{make_float4(1, 0, 0, 0), make_float4(0, 1, 0, 0), make_float4(0, 0, 1, 0), make_float4(0, 0, 0, 1)}};
}

inline float4 operator*(const Transform4 &lhs, const float4 &rhs)
{
    return make_float4(dot(lhs.m[0], rhs), dot(lhs.m[1], rhs), dot(lhs.m[2], rhs), dot(lhs.m[3], rhs));
}
} // namespace otk

#endif
