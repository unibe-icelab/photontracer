// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

// Where a ray continues after it hit a triangle. The hit point is rebuilt from the
// barycentric coordinates, which keeps it on the triangle, and the spawn points are
// moved along the normal by more than the rounding error of that reconstruction, so
// the next trace does not hit the same triangle again. The OptiX backend uses the
// directed-rounding version of this from the OptiX Toolkit.

#pragma once

#include <cfloat>
#include <cmath>

#include "portable_math.h"

struct SpawnPoint
{
    float3 position; // on the triangle
    float3 normal;   // unit length, from the vertex order
    float3 front;    // outside, on the side of the normal
    float3 back;     // inside
};

/// Multiple of the rounding error of the hit point that the spawn points are moved away
constexpr float SPAWN_OFFSET_SCALE = 32.0f;

inline float maxAbs(float3 v)
{
    return std::fmax(std::fabs(v.x), std::fmax(std::fabs(v.y), std::fabs(v.z)));
}

/// @param u, v Barycentric coordinates of the hit, weights of v1 and v2
inline SpawnPoint triangleSpawnPoint(float3 v0, float3 v1, float3 v2, float u, float v)
{
    const float3 e1 = v1 - v0;
    const float3 e2 = v2 - v0;

    SpawnPoint spawn;
    spawn.position = v0 + e1 * u + e2 * v;
    spawn.normal = otk::normalize(otk::cross(e1, e2));

    // The magnitudes of the terms the position is summed from bound its rounding error. Planes
    // that are exactly representable, like z = 0, have none, so the size of the triangle and its
    // distance from the origin set a floor; a ray starting exactly on the plane would hit it again.
    const float errorX = std::fabs(v0.x) + std::fabs(e1.x * u) + std::fabs(e2.x * v);
    const float errorY = std::fabs(v0.y) + std::fabs(e1.y * u) + std::fabs(e2.y * v);
    const float errorZ = std::fabs(v0.z) + std::fabs(e1.z * u) + std::fabs(e2.z * v);
    const float floor = std::fmax(std::fmax(errorX, errorY), std::fmax(errorZ, std::fmax(maxAbs(e1), maxAbs(e2))));
    const float offset = SPAWN_OFFSET_SCALE * FLT_EPSILON *
                         (std::fabs(spawn.normal.x) * errorX + std::fabs(spawn.normal.y) * errorY +
                          std::fabs(spawn.normal.z) * errorZ + floor);

    spawn.front = spawn.position + spawn.normal * offset;
    spawn.back = spawn.position - spawn.normal * offset;
    return spawn;
}
