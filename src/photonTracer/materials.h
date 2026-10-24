// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <cstdint>

enum MaterialType
{
    DIFFUSE,
    REFRACTIVE,
    VOLUME_SCATTERING,
    REFLECTIVE,
};

struct RefractiveIndex
{
    float r;
    float i;
};

union MaterialProperties
{
    struct Diffuse
    {
        float albedo; // Albedo for Diffuse material
    } diffuse;

    struct Refractive
    {
        RefractiveIndex refractiveIndex; // Refractive index for the material
    } refractive;

    struct VolumeScattering
    {
        RefractiveIndex refractiveIndex; // Refractive index for the medium (absorption coefficient in k)
        float scatteringCoefficient;     // Scattering coefficient for the medium
        float asymetryParameter;         // Asymmetry parameter for the Henyey-Greenstein phase function
    } volumeScattering;

    struct Reflective
    {
        float reflectivity; // Reflectivity for the material
        float fuzziness;    // Fuzziness for the material
    } reflective;
};

struct Material
{
    MaterialType type;             // Type of the material
    MaterialProperties properties; // Properties of the material
};

constexpr uint32_t MAX_MATERIALS = 16; // the medium index is stored in 4 bits

struct HitGroupData
{
    Material materials[MAX_MATERIALS]; // Array of refractive indices for each material
    float wavelengthUm;
};
