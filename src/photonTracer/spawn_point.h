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

/// Point on a surface with its unit normal, and the distance along the normal that is safe to move away
struct SpawnPoint
{
    float3 position;
    float3 normal;
    float offset;

    float3 front() const { return position + normal * offset; } ///< outside, on the side of the normal
    float3 back() const { return position - normal * offset; }  ///< inside
};

/// Affine transform of an instance: `matrix` is 3x4 row-major, `inverse` the inverse of its 3x3 part
struct InstanceTransform
{
    float matrix[12];
    float inverse[9];
};

/// Multiple of the rounding error of a position that the spawn points are moved away. The tests
/// fail below 2. A larger offset loses the events of rays that cross a surface closer than the
/// offset: it lowers the mean scattering count of a particle cloud by 0.15% at 4 and 1% at 32.
constexpr float SPAWN_OFFSET_SCALE = 4.0f;

/// Weight of the size of the triangle and the distance from the origin in the offset
constexpr float SPAWN_FLOOR_WEIGHT = 0.25f;

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
    spawn.offset = SPAWN_OFFSET_SCALE * FLT_EPSILON *
                   (std::fabs(spawn.normal.x) * errorX + std::fabs(spawn.normal.y) * errorY +
                    std::fabs(spawn.normal.z) * errorZ + SPAWN_FLOOR_WEIGHT * floor);
    return spawn;
}

/// Moves a spawn point from the space of an instance to the space the instance is placed in
inline SpawnPoint transformSpawnPoint(const SpawnPoint &spawn, const InstanceTransform &transform)
{
    const float *m = transform.matrix;
    const float *inv = transform.inverse;
    const float3 p = spawn.position;

    SpawnPoint result;
    result.position = make_float3(m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
                                  m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
                                  m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);

    // Normals transform with the inverse transpose
    const float3 n = spawn.normal;
    const float3 normal = make_float3(inv[0] * n.x + inv[3] * n.y + inv[6] * n.z,
                                      inv[1] * n.x + inv[4] * n.y + inv[7] * n.z,
                                      inv[2] * n.x + inv[5] * n.y + inv[8] * n.z);
    const float normalLength = otk::length(normal);
    result.normal = normal / normalLength;

    // A displacement of `offset` along the old normal is `offset / normalLength` along the new one
    const float transported = spawn.offset / normalLength;

    const float errorX = std::fabs(m[0] * p.x) + std::fabs(m[1] * p.y) + std::fabs(m[2] * p.z) + std::fabs(m[3]);
    const float errorY = std::fabs(m[4] * p.x) + std::fabs(m[5] * p.y) + std::fabs(m[6] * p.z) + std::fabs(m[7]);
    const float errorZ = std::fabs(m[8] * p.x) + std::fabs(m[9] * p.y) + std::fabs(m[10] * p.z) + std::fabs(m[11]);
    result.offset = transported + SPAWN_OFFSET_SCALE * FLT_EPSILON *
                                      (std::fabs(result.normal.x) * errorX + std::fabs(result.normal.y) * errorY +
                                       std::fabs(result.normal.z) * errorZ);
    return result;
}

/// Builds the transform of an instance from its 3x4 row-major matrix; false if the matrix is singular
inline bool makeInstanceTransform(const float *matrix, InstanceTransform &transform)
{
    for (int i = 0; i < 12; ++i)
    {
        transform.matrix[i] = matrix[i];
    }

    // Inverse of the 3x3 part from its cofactors, in double precision
    double a[9];
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            a[3 * row + column] = matrix[4 * row + column];
        }
    }
    const double c00 = a[4] * a[8] - a[5] * a[7];
    const double c01 = a[5] * a[6] - a[3] * a[8];
    const double c02 = a[3] * a[7] - a[4] * a[6];
    const double determinant = a[0] * c00 + a[1] * c01 + a[2] * c02;
    if (determinant == 0.0 || !std::isfinite(determinant))
    {
        return false;
    }
    const double inv[9] = {c00, a[2] * a[7] - a[1] * a[8], a[1] * a[5] - a[2] * a[4],
                           c01, a[0] * a[8] - a[2] * a[6], a[2] * a[3] - a[0] * a[5],
                           c02, a[1] * a[6] - a[0] * a[7], a[0] * a[4] - a[1] * a[3]};
    for (int i = 0; i < 9; ++i)
    {
        transform.inverse[i] = static_cast<float>(inv[i] / determinant);
    }
    return true;
}
