// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/Transform4.h>
#include "complex_f.cuh"

#ifndef FRESNEL_H
#define FRESNEL_H

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

#if defined(__CUDACC__)

/**
 * @brief Calculate the absorption coefficient
 *
 * @param[in] k The absorption index of the medium
 * @param[in] wavelengthUm The wavelength of the ray
 * @param[in] lengthScale Scaling factor to conver um to the length unit of the sim
 *
 * @return The absorption coefficient
 */
__forceinline__ __host__ __device__ float absorptionCoefficientFromk(float const &k, float const &wavelengthUm, float const &lengthScale)
{
    return (4.0f * M_PIf * k) / (wavelengthUm * lengthScale);
}

/**
 * @brief Calculate the distance travelled by a ray and whether it is absorbed
 *
 * @param[in] maxDistance The maximum distance the ray can travel in milli meters
 * @param[in] wavelengthUm The wavelength of the ray in micrometers
 * @param[in] k The absorption coefficient of the medium
 * @param[in] randomSample A random number in [0, 1)
 *
 * @param[out] travelledDistance The distance travelled by the ray in milli meters
 *
 * @return Ray is absorbed
 */
__forceinline__ __host__ __device__ bool calculateAbsorption(
    const float &maxDistance, const float &wavelengthUm, const float &lengthScale, const float &k, const float &randomSample, float &travelledDistance)
{
    // travelledDistance = maxDistance;
    // return false;
    if (k == 0.0f)
    {
        travelledDistance = maxDistance;
        return false;
    }
    travelledDistance = - logf(randomSample) / absorptionCoefficientFromk(k, wavelengthUm, lengthScale);
    if (travelledDistance < maxDistance)
    {
        return true;
    }
    else
    {
        travelledDistance = maxDistance;
        return false;
    }
}


/**
 * @brief Calculate the distance travelled by a ray and whether it is absorbed/scattered
 *
 * @param[in] maxDistance The maximum distance the ray can travel in milli meters
 * @param[in] extinctionCoefficient The extinction coefficient of the medium
 * @param[in] randomSample A random number in [0, 1)
 *
 * @param[out] travelledDistance The distance travelled by the ray in milli meters
 *
 * @return Ray is absorbed/scattered
 */
__forceinline__ __host__ __device__ bool calculateDistance(
    const float &maxDistance, const float &extinctionCoefficient, const float &randomSample, float &travelledDistance)
{
    // travelledDistance = maxDistance;
    // return false;
    if (extinctionCoefficient == 0.0f)
    {
        travelledDistance = maxDistance;
        return false;
    }
    travelledDistance = - logf(randomSample) / extinctionCoefficient;
    if (travelledDistance < maxDistance)
    {
        return true;
    }
    else
    {
        travelledDistance = maxDistance;
        return false;
    }
}

__forceinline__ __host__ __device__ float3 henyeyGreensteinDirection(float3 const& directionIn, float g, float rndSample1, float rndSample2){
    float cosTheta;
    if (fabsf(g) < 1e-6f) {
        cosTheta = 1.0f - 2.0f * rndSample1;
    } else {
        float sqrTerm = (1.0f - g * g) / (1.0f - g + 2.0f * g * rndSample1);
        cosTheta = (1.0f + g * g - sqrTerm * sqrTerm) / (2.0f * g);
    }
    cosTheta = fminf(1.0f, fmaxf(-1.0f, cosTheta));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    float phi = 2.0f * M_PIf * rndSample2;
    float cosPhi = cosf(phi);
    float sinPhi = sinf(phi);

    // Create an orthonormal basis
    float3 w = otk::normalize(directionIn);
    float3 u = otk::normalize(otk::cross(fabsf(w.x) > 0.1f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f), w));
    float3 v = otk::cross(w, u);

    // Convert spherical coordinates to Cartesian coordinates
    float3 directionOut = sinTheta * cosPhi * u + sinTheta * sinPhi * v + cosTheta * w;
    return otk::normalize(directionOut);
}


/**
 * @brief Calculate the angle of rotation of the stokes vector
 *
 * @param[in] vector1 The normal of the scattering plane
 * @param[in] vector2 The normal of the scattering plane of the last interaction
 *
 * @return The angle of rotation of the stokes vector in radians
 */
__forceinline__ __host__ __device__ float unsignedAngleBetweenVectors(
    const float3 vector1, const float3 vector2)
{
    float cosThetha = otk::dot(vector1, vector2);
    cosThetha = fminf(cosThetha, 1.0f);
    cosThetha = fmaxf(cosThetha, -1.0f);
    return acosf(cosThetha);
}

/**
 * @brief Calculate the signed rotation angle between two vectors about an axis
 *
 * @param[in] axis The axis of rotation as a unit vector
 * @param[in] vector1 The first vector as a unit vector
 * @param[in] vector2 The second vector as a unit vector
 *
 * @return The signed angle of rotation from vector1 to vector2 about the axis in radians
 */
__forceinline__ __host__ __device__ float signedRotationAboutAxis(
    const float3 &axis,
    const float3 &vector1,
    const float3 &vector2)
{
    // q_old, q_new must be unit and ⟂ k
    float3 c = otk::cross(vector1, vector2);
    float s = otk::dot(axis, c);           // sin component (signed)
    float c0 = otk::dot(vector1, vector2); // cos component
    return atan2f(s, c0);
}

/**
 * @brief Calculate the reflectivity using Schlick's approximation (not used in the standard implementation)
 *
 * @param[in] cosTheta The cosine of the angle of incidence
 * @param[in] na The refractive index of the medium the incident ray is in
 * @param[in] nb The refractive index of the medium the refracted ray will be in
 *
 * @return The reflectivity
 */
__forceinline__ __host__ __device__ float calculateReflectivitySchlick(float cosTheta, float na, float nb)
{
    float r0 = (na - nb) / (na + nb);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosTheta), 5.0f);
}

/**
 * @brief Calculate the angle of incidence and refraction, and whether total internal reflection occurs
 *
 * @param[in] direction The direction of the incident ray, as a unit vector
 * @param[in] normal The normal of the surface, as a unit vector; pointing in the opposite direction of the incident ray
 * @param[in] na The refractive index of the medium the incident ray is in
 * @param[in] nb The refractive index of the medium the refracted ray will be in
 *
 * @param[out] cosTheta Angle of incidence
 * @param[out] cosThetaPrime Angle of refraction
 * @param[out] totalInternalReflection Whether total internal reflection occurs
 */
__forceinline__ __host__ __device__ void calculateSnellsLaw(
    const float3 &direction, const float3 &normal, const float &na, const float &nb,
    float &cosTheta, float &cosThetaPrime, bool &totalInternalReflection)
{

    // assumes that the normal is pointing in the opposite direction of the incident ray
    cosTheta = -otk::dot(direction, normal);
    float eta = na / nb;

    float radicand = 1.0f - eta * eta * (1.0f - cosTheta * cosTheta);

    if (radicand > 0.f)
    {
        cosThetaPrime = sqrtf(radicand);
        totalInternalReflection = false;
    }
    else
    {
        cosThetaPrime = 0.f;
        totalInternalReflection = true;
    }
}

/**
 * @brief Calculate a random lambertian direction based on two random numbers and a normal,
 * by sampling uniformly over the unit sphere and adding it to the normal vector
 *
 * @param[in] randomNumber1 A random number between 0 and 1
 * @param[in] randomNumber2 A random number between 0 and 1
 * @param[in] normal The normal of the surface as a unit vector
 *
 * @return A random lambertian direction as a unit vector
 */
__forceinline__ __host__ __device__ float3 calculateLamberianDirection(
    float randomNumber1, float randomNumber2, const float3 &normal)
{
    float theta = 2.0f * M_PIf * randomNumber1;     // Azimuthal angle
    float phi = acosf(2.0f * randomNumber2 - 1.0f); // Polar angle

    float x = sinf(phi) * cosf(theta);
    float y = sinf(phi) * sinf(theta);
    float z = cosf(phi);

    float3 direction = normal + make_float3(x, y, z);
    return otk::normalize(direction);
}

/**
 * @brief Calculate the direction of the refracted ray
 *
 * @param[in] direction The direction of the incident ray
 * @param[in] normal The normal of the surface as a unit vector; pointing in the opposite direction of the incident ray
 * @param[in] cosTheta Angle of incidence
 * @param[in] cosThetaPrime Angle of refraction
 * @param[in] na The refractive index of the medium the incident ray is in
 * @param[in] nb The refractive index of the medium the refracted ray will be in
 *
 * @return The direction of the refracted ray
 */
__forceinline__ __host__ __device__ float3 calculateRefractedDirection(
    const float3 &direction, const float3 &normal,
    const float &cosTheta, const float &cosThetaPrime,
    const float &na, const float &nb)
{
    // assumes that the normal is pointing in the opposite direction of the incident ray
    float eta = na / nb;
    float3 refracted_direction = eta * direction + (eta * cosTheta - cosThetaPrime) * normal;
    return refracted_direction;
}

/**
 * @brief Calculate the direction of the reflected ray
 *
 * @param[in] direction The direction of the incident ray as a unit vector
 * @param[in] normal The normal of the surface as a unit vector; pointing in the opposite direction of the incident ray
 *
 * @return The direction of the reflected ray
 */
__forceinline__ __host__ __device__ float3 calculateReflectedDirection(
    const float3 &direction, const float3 &normal)
{
    return direction - 2.0f * otk::dot(direction, normal) * normal;
}

/**
 * @brief Calculate the normal axis of the scattering plane
 *
 * @param[in] k The direction of the incident ray as a unit vector
 * @param[in] surfaceNormal The normal of the surface as a unit vector; pointing in the opposite direction of the incident ray
 *
 * @param[out] scatteringPlaneNormal The normal of the scattering plane as a unit vector or [NAN, NAN, NAN] if the ray is (anti-)parallel to the normal
 */
__forceinline__ __host__ __device__ void scatteringPlaneNormalAxis(
    const float3 &k, const float3 &surfaceNormal,
    float3 &scatteringPlaneNormal)
{
    // Robust, deterministic normal to the scattering plane, which aligns with -Q axis ⟂ k
    float3 crossVector = otk::cross(surfaceNormal, k);
    float magnitudeSquared = otk::dot(crossVector, crossVector);
    if (magnitudeSquared < 1e-12f)
    {
        scatteringPlaneNormal = make_float3(NAN, NAN, NAN);
        return;
    }
    scatteringPlaneNormal = crossVector * rsqrtf(magnitudeSquared);
    // -Q axis/⟂ k, pointing along n_plane × k
}

/**
 * @brief Calculate the rotation matrix for rotating the stokes vector
 *
 * @param[in] omega The angle of rotation in radians
 *
 * @return The rotation matrix as a Transform4
 */
__forceinline__ __host__ __device__ otk::Transform4 calculateRotationMatrix(float omega)
{
    if (omega == 0.0f)
    {
        return otk::identity();
    }
    else
    {
        float cos2omega = cosf(2 * omega);
        float sin2omega = sinf(2 * omega);

        otk::Transform4 rot;

        rot.m[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
        rot.m[1] = make_float4(0.0f, cos2omega, sin2omega, 0.0f);
        rot.m[2] = make_float4(0.0f, -sin2omega, cos2omega, 0.0f);
        rot.m[3] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

        return rot;
    }
}

/**
 * @brief Calculate the Muller matrix for total internal reflection
 *
 * @param[in] cosThetaA The cosine of the angle of incidence
 * @param[in] na The refractive index of the medium the incident ray is in
 * @param[in] nb The refractive index of the medium the refracted ray will be in
 *
 * @return The Muller matrix as a Transform4
 */
__forceinline__ __host__ __device__ otk::Transform4 calculateMullerTotalReflection(
    const float &cosThetaA, const float &na, const float &nb)
{
    // Light Scattering Reviews: Light scattering from particulate surfaces in  geometrical optics approximation, Grynko & Skuratov 2008

    float eta_ = nb / na;
    float eta_Sq = eta_ * eta_;

    float thetaA = acosf(cosThetaA);
    float sinThetaA = sinf(thetaA);
    float sinThetaASq = sinThetaA * sinThetaA;
    float rootArg = sinThetaASq - eta_Sq;
    float rootArgClamped = fmaxf(rootArg, 0.0f);
    float root = sqrt(rootArgClamped);

    float tanDeltaP_2 = -root / (eta_Sq * cosThetaA);
    float tanDeltaS_2 = -root / cosThetaA;

    float deltaP = atanf(tanDeltaP_2) * 2.0f;
    float deltaS = atanf(tanDeltaS_2) * 2.0f;

    float deltaDiff = deltaP - deltaS;

    float cosDeltaDiff = cosf(deltaDiff);
    float sinDeltaDiff = sinf(deltaDiff);

    otk::Transform4 rTot;
    rTot.m[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    rTot.m[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    rTot.m[2] = make_float4(0.0f, 0.0f, cosDeltaDiff, sinDeltaDiff);
    rTot.m[3] = make_float4(0.0f, 0.0f, -sinDeltaDiff, cosDeltaDiff);
    return rTot;
}

/**
 * @brief Calculate the Muller matrix for reflection
 *
 * @param[in] cosThetaA The cosine of the angle of incidence
 * @param[in] cosThetaB The cosine of the angle of refraction
 * @param[in] na The refractive index of the medium the incident ray is in
 * @param[in] nb The refractive index of the medium the refracted ray will be in
 *
 * @return The Muller matrix as a Transform4
 */
__forceinline__ __host__ __device__ otk::Transform4 calculateMullerReflection(
    const float &cosThetaA, const float &cosThetaB, const float &na, const float &nb)
{
    // Light Scattering Reviews: Light scattering from particulate surfaces in  geometrical optics approximation, Grynko & Skuratov 2008
    float eta = na / nb;

    float rS = (eta * cosThetaA - cosThetaB) / (eta * cosThetaA + cosThetaB);
    float rS2 = rS * rS;

    float rP = (cosThetaA - eta * cosThetaB) / (cosThetaA + eta * cosThetaB);
    float rP2 = rP * rP;

    float rSrP = rP * rS;

    float rSrPPNorm = 0.5f * (rP2 + rS2);
    float rSrPMNorm = 0.5f * (rP2 - rS2);

    otk::Transform4 r;

    r.m[0] = make_float4(rSrPPNorm, rSrPMNorm, 0.0f, 0.0f);
    r.m[1] = make_float4(rSrPMNorm, rSrPPNorm, 0.0f, 0.0f);
    r.m[2] = make_float4(0.0f, 0.0f, rSrP, 0.0f);
    r.m[3] = make_float4(0.0f, 0.0f, 0.0f, rSrP);

    return r;
}

/**
 * @brief Calculate the Muller matrix for transmission
 *
 * @param[in] cosThetaA The cosine of the angle of incidence
 * @param[in] cosThetaB The cosine of the angle of refraction
 * @param[in] na The refractive index of the medium the incident ray is in
 * @param[in] nb The refractive index of the medium the refracted ray will be in
 *
 * @return The Muller matrix as a Transform4
 */
__forceinline__ __host__ __device__ otk::Transform4 calculateMullerTransmission(
    const float &cosThetaA, const float &cosThetaB, const float &na, const float &nb)
{
    // Light Scattering Reviews: Light scattering from particulate surfaces in  geometrical optics approximation, Grynko & Skuratov 2008

    float eta = na / nb;

    float ts = 2.0f * eta * cosThetaA / (eta * cosThetaA + cosThetaB);
    float ts2 = ts * ts;

    float tp = 2.0f * eta * cosThetaA / (cosThetaA + eta * cosThetaB);
    float tp2 = tp * tp;

    float f = cosThetaB / (2.0f * cosThetaA * eta);

    float tsTpNorm = 2 * ts * tp * f;

    float tsTpPNorm = (tp2 + ts2) * f;
    float tsTpMNorm = (tp2 - ts2) * f;

    otk::Transform4 t;
    t.m[0] = make_float4(tsTpPNorm, tsTpMNorm, 0.0f, 0.0f);
    t.m[1] = make_float4(tsTpMNorm, tsTpPNorm, 0.0f, 0.0f);
    t.m[2] = make_float4(0.0f, 0.0f, tsTpNorm, 0.0f);
    t.m[3] = make_float4(0.0f, 0.0f, 0.0f, tsTpNorm);

    return t;
}


/**
 * @brief Calculate the Fresnel interaction of a ray with a surface
 *
 * @param[in,out] stokesVector The stokes vector of the ray
 * @param[in,out] qMinusAxis The -Q axis of the ray
 * @param[in,out] direction The direction of the ray
 * @param[in] surfaceNormal The normal of the surface as a unit vector; pointing in the opposite direction of the incident ray
 * @param[in] nA The refractive index of the medium the incident ray is in
 * @param[in] nB The refractive index of the medium the refracted ray will be in
 * @param[in] randomSample A random number between 0 and 1
 */
__host__ __device__ bool calculateFresnelInteraction(
    float4 &stokesVector, float3 &qMinusAxis,
    float3 &direction, const float3 &surfaceNormal,
    const float &nA, const float &nB, const float &randomSample)
{

    // calculate snells law to get refracted angle
    float cosTheta, cosThetaPrime;
    bool totalInternalReflection;
    calculateSnellsLaw(direction, surfaceNormal, nA, nB, cosTheta, cosThetaPrime, totalInternalReflection);

    // rotate stokes vector
    float3 nextQMinusAxis;
    scatteringPlaneNormalAxis(direction, surfaceNormal, nextQMinusAxis);

    // only update qMinusAxis, if direction is not (anti-)parallel to surface normal
    if (!isnan(nextQMinusAxis.x))
    {
        float omega = signedRotationAboutAxis(direction, qMinusAxis, nextQMinusAxis);
        qMinusAxis = nextQMinusAxis;
        otk::Transform4 stokesRotation = calculateRotationMatrix(omega);
        stokesVector = stokesRotation * stokesVector;
    }

    otk::Transform4 mullerMatrix;
    if (totalInternalReflection)
    {
        mullerMatrix = calculateMullerTotalReflection(cosTheta, nA, nB);
        direction = calculateReflectedDirection(direction, surfaceNormal);
        stokesVector = mullerMatrix * stokesVector;
        return true;
    }
    else
    {
        // calculate transmissivity from stokes vector
        // transmission first because it is more likely than reflection
        mullerMatrix = calculateMullerTransmission(cosTheta, cosThetaPrime, nA, nB);
        float4 stokesVectorTransmission = mullerMatrix * stokesVector;
        float transmissivity = stokesVectorTransmission.x;

        if (randomSample < transmissivity)
        {
            stokesVector = stokesVectorTransmission / transmissivity;
            direction = calculateRefractedDirection(direction, surfaceNormal, cosTheta, cosThetaPrime, nA, nB);
            return false;
        }
        else
        {
            mullerMatrix = calculateMullerReflection(cosTheta, cosThetaPrime, nA, nB);
            stokesVector = mullerMatrix * stokesVector;
            stokesVector = stokesVector / stokesVector.x;
            direction = calculateReflectedDirection(direction, surfaceNormal);
            return true;
        }
    }
}


static __forceinline__ __host__ __device__ uint32_t packMediumHistory(uint32_t mediumHistory[8])
{
    uint32_t packedMediumHistory = 0;
    for (int i = 0; i < 8; ++i)
    {
        packedMediumHistory |= (mediumHistory[i] & 0xF) << (i * 4); // Each layer is 4 bits
    }
    return packedMediumHistory;
}

static __forceinline__ __host__ __device__ void unpackMediumHistory(uint32_t packedMediumHistory, uint32_t mediumHistory[8])
{
    for (int i = 0; i < 8; ++i)
    {
        mediumHistory[i] = (packedMediumHistory >> (i * 4)) & 0xF; // Extract each layer (4 bits)
    }
}

static __forceinline__ __host__ __device__ uint32_t getMediumFromPacked(uint32_t packedMediumHistory, uint32_t index)
{
    if (index >= 8)
    {
        return 0;
    }
    return (packedMediumHistory >> (index * 4)) & 0xF;
}

static __forceinline__ __host__ __device__ bool appendMediumPacked(uint32_t medium, uint32_t &mediumHistorySize, uint32_t &packedMediumHistory)
{
    if (mediumHistorySize >= 8)
    {
        return false;
    }
    packedMediumHistory |= (medium & 0xF) << (mediumHistorySize * 4);
    mediumHistorySize++;
    return true;
}

static __forceinline__ __host__ __device__ bool removeLastOccurencePacked(uint32_t medium, uint32_t &mediumHistorySize, uint32_t &packedMediumHistory)
{
    if (mediumHistorySize == 0)
    {
        return false;
    }
    for (int i = static_cast<int>(mediumHistorySize) - 1; i >= 0; --i)
    {
        const uint32_t entry = (packedMediumHistory >> (i * 4)) & 0xF;
        if (entry == medium)
        {
            for (int j = i; j < static_cast<int>(mediumHistorySize) - 1; ++j)
            {
                const uint32_t nextEntry = (packedMediumHistory >> ((j + 1) * 4)) & 0xF;
                packedMediumHistory &= ~(0xF << (j * 4));
                packedMediumHistory |= (nextEntry & 0xF) << (j * 4);
            }
            packedMediumHistory &= ~(0xF << ((mediumHistorySize - 1) * 4));
            mediumHistorySize--;
            return true;
        }
    }
    return false;
}

/**
 * @brief Calculate the Muller matrix for reflection with complex refractive indices
 *
 * @param[in] cosThetaA The cosine of the angle of incidence
 * @param[in] cosThetaB The cosine of the angle of refraction
 * @param[in] na The refractive index of the medium the incident ray is in
 * @param[in] nb The refractive index of the medium the refracted ray will be in
 *
 * @return The Muller matrix as a Transform4
 */
__forceinline__ __host__ __device__ otk::Transform4 calculateMullerReflection(
    const float &cosThetaI, const float &cosThetaT, 
    const Complexf &na, const Complexf &nb)
{
    // Muinonen et al. 1996, Väsanien et al. 2018 (thetha = psi, homogeneous wave treatment)
    Complexf ki = na * cosThetaI;
    Complexf kt = nb * cosThetaT;

    Complexf div1 = nb * nb * ki + na * na * kt;
    Complexf div2 = ki + kt;
    float div_norm1 = norm(div1);
    float div_norm2 = norm(div2);

    if ((div_norm1 < 1e-12) || (div_norm2 < 1e-12))
    {
        printf("Warning: Division by zero in calculateMullerReflection with complex refractive indices. Fallback to identity matrix.\n");
        // avoid division by zero
        otk::Transform4 r;
        r.m[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
        r.m[1] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
        r.m[2] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
        r.m[3] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        return r;
    }

    Complexf rp = (nb * nb * ki - na * na * kt) * conj(div1) / (div_norm1);
    Complexf rs = (ki - kt) * conj(div2) / (div_norm2);

    float rsrs_ = norm(rs);
    float rprp_ = norm(rp);

    float rp2prs2_2 = 0.5f * (rprp_ + rsrs_);
    float rp2mrs2_2 = 0.5f * (rprp_ - rsrs_);

    Complexf rprs = rp * conj(rs);

    otk::Transform4 r;
    r.m[0] = make_float4(rp2prs2_2, rp2mrs2_2, 0.0f,           0.0f);
    r.m[1] = make_float4(rp2mrs2_2, rp2prs2_2, 0.0f,           0.0f);
    r.m[2] = make_float4(0.0f,      0.0f,      rprs.real(),  rprs.imag());
    r.m[3] = make_float4(0.0f,      0.0f,      -rprs.imag(), rprs.real());

    return r;
}

/**
 * @brief Calculate the Muller matrix for transmission with complex refractive indices
 *
 * @param[in] cosThetaA The cosine of the angle of incidence
 * @param[in] cosThetaB The cosine of the angle of refraction
 * @param[in] na The refractive index of the medium the incident ray is in
 * @param[in] nb The refractive index of the medium the refracted ray will be in
 *
 * @return The Muller matrix as a Transform4
 */
__forceinline__ __host__ __device__ otk::Transform4 calculateMullerTransmission(
    const float &cosThetaI, const float &cosThetaT, const Complexf &na, const Complexf &nb)
{
    // Muinonen et al. 1996, Väsanien et al. 2018 (thetha = psi, homogeneous wave treatment)
    Complexf ki = na * cosThetaI;
    Complexf kt = nb * cosThetaT;

    Complexf tp = 2.0f * na * nb * ki / (nb * nb * ki + na * na * kt);
    Complexf ts = 2.0f * ki / (ki + kt);

    float tsts_ = norm(ts);
    float tptp_ = norm(tp);

    float tp2pts2_2 = 0.5f * (tptp_ + tsts_);
    float tp2mts2_2 = 0.5f * (tptp_ - tsts_);

    Complexf tpts = tp * conj(ts);

    otk::Transform4 t;
    t.m[0] = make_float4(tp2pts2_2, tp2mts2_2, 0.0f,           0.0f);
    t.m[1] = make_float4(tp2mts2_2, tp2pts2_2, 0.0f,           0.0f);
    t.m[2] = make_float4(0.0f,      0.0f,      tpts.real(),    tpts.imag());
    t.m[3] = make_float4(0.0f,      0.0f,      -tpts.imag(),   tpts.real());

    return t;
}

/**
 * @brief Calculate the Fresnel interaction of a ray with a surface for absorbing media
 *
 * @param[in,out] stokesVector The stokes vector of the ray
 * @param[in,out] qMinusAxis The -Q axis of the ray
 * @param[in,out] direction The direction of the ray
 * @param[in] surfaceNormal The normal of the surface as a unit vector; pointing in the opposite direction of the incident ray
 * @param[in] nA The refractive index of the medium the incident ray is in
 * @param[in] nB The refractive index of the medium the refracted ray will be in
 * @param[in] randomSample A random number between 0 and 1
 */
__host__ __device__ bool calculateFresnelInteraction(
    float4 &stokesVector, float3 &qMinusAxis,
    float3 &direction, const float3 &surfaceNormal,
    const Complexf &nA, const Complexf &nB, const float &randomSample)
{

    // calculate snells law to get refracted angle
    float cosTheta, cosThetaPrime;
    bool totalInternalReflection;
    calculateSnellsLaw(direction, surfaceNormal, nA.real(), nB.real(), cosTheta, cosThetaPrime, totalInternalReflection);

    // rotate stokes vector
    float3 nextQMinusAxis;
    scatteringPlaneNormalAxis(direction, surfaceNormal, nextQMinusAxis);

    // only update qMinusAxis, if direction is not (anti-)parallel to surface normal
    if (!isnan(nextQMinusAxis.x))
    {
        float omega = signedRotationAboutAxis(direction, qMinusAxis, nextQMinusAxis);
        qMinusAxis = nextQMinusAxis;
        otk::Transform4 stokesRotation = calculateRotationMatrix(omega);
        float4 stokesVectorRot = stokesRotation * stokesVector;
        if ((isnan(stokesVectorRot.x) || isnan(stokesVectorRot.y) || isnan(stokesVectorRot.z) || isnan(stokesVectorRot.w)) && !(isnan(stokesVector.x) || isnan(stokesVector.y) || isnan(stokesVector.z) || isnan(stokesVector.w)))
        {
            printf("NaN in stokes vector after rotation with complex refractive indices\n");
            printf("omega: %f\n", omega);
            printf("stokesVector before: (%f, %f, %f, %f)\n", stokesVector.x, stokesVector.y, stokesVector.z, stokesVector.w);
            printf("stokesVectorRot: (%f, %f, %f, %f)\n", stokesVectorRot.x, stokesVectorRot.y, stokesVectorRot.z, stokesVectorRot.w);
        }
        stokesVector = stokesVectorRot;
        
    }

    otk::Transform4 mullerMatrix;
    if (totalInternalReflection)
    {
        mullerMatrix = calculateMullerTotalReflection(cosTheta, nA.real(), nB.real());
        direction = calculateReflectedDirection(direction, surfaceNormal);
        float4 stokesVectorReflection = mullerMatrix * stokesVector;
        if ((isnan(stokesVectorReflection.x) || isnan(stokesVectorReflection.y) || isnan(stokesVectorReflection.z) || isnan(stokesVectorReflection.w)) && !(isnan(stokesVector.x) || isnan(stokesVector.y) || isnan(stokesVector.z) || isnan(stokesVector.w)))
        {
            printf("NaN in stokes vector after total internal reflection with complex refractive indices\n");
            printf("cosTheta: %f, nA: (%f, %f), nB: (%f, %f)\n", cosTheta, nA.real(), nA.imag(), nB.real(), nB.imag());
            printf("stokesVector before: (%f, %f, %f, %f)\n", stokesVector.x, stokesVector.y, stokesVector.z, stokesVector.w);
            printf("mullerMatrix: \n");
            for (int i = 0; i < 4; ++i)
            {
                printf("(%f, %f, %f, %f)\n", mullerMatrix.m[i].x, mullerMatrix.m[i].y, mullerMatrix.m[i].z, mullerMatrix.m[i].w);
            }
            printf("stokesVectorReflection: (%f, %f, %f, %f)\n", stokesVectorReflection.x, stokesVectorReflection.y, stokesVectorReflection.z, stokesVectorReflection.w);
        }
        stokesVector = stokesVectorReflection;
        return true;
    }
    else
    {
        // calculate transmissivity from stokes vector
        // transmission first because it is more likely than reflection
        mullerMatrix = calculateMullerReflection(cosTheta, cosThetaPrime, nA, nB);
        float4 stokesVectorReflection = mullerMatrix * stokesVector;
        float reflectivity = stokesVectorReflection.x;

        if (randomSample <= reflectivity)
        {
            stokesVectorReflection = stokesVectorReflection / stokesVectorReflection.x;
            direction = calculateReflectedDirection(direction, surfaceNormal);
            if ((isnan(stokesVectorReflection.x) || isnan(stokesVectorReflection.y) || isnan(stokesVectorReflection.z) || isnan(stokesVectorReflection.w)) && !(isnan(stokesVector.x) || isnan(stokesVector.y) || isnan(stokesVector.z) || isnan(stokesVector.w)))
            {
                printf("NaN in stokes vector after reflection with complex refractive indices\n");
                printf("cosTheta: %f, cosThetaPrime: %f, nA: (%f, %f), nB: (%f, %f)\n", cosTheta, cosThetaPrime, nA.real(), nA.imag(), nB.real(), nB.imag());
                printf("stokesVector before: (%f, %f, %f, %f)\n", stokesVector.x, stokesVector.y, stokesVector.z, stokesVector.w);
                printf("mullerMatrix: \n");
                for (int i = 0; i < 4; ++i)
                {
                    printf("(%f, %f, %f, %f)\n", mullerMatrix.m[i].x, mullerMatrix.m[i].y, mullerMatrix.m[i].z, mullerMatrix.m[i].w);
                }
                printf("stokesVectorReflection: (%f, %f, %f, %f)\n", stokesVectorReflection.x, stokesVectorReflection.y, stokesVectorReflection.z, stokesVectorReflection.w);
            }
            stokesVector = stokesVectorReflection;
            return true;
        }
        else
        {
            mullerMatrix = calculateMullerTransmission(cosTheta, cosThetaPrime, nA, nB);
            float4 stokesVectorTransmission = mullerMatrix * stokesVector;
            stokesVectorTransmission = stokesVectorTransmission / stokesVectorTransmission.x;
            direction = calculateRefractedDirection(direction, surfaceNormal, cosTheta, cosThetaPrime, nA.real(), nB.real());
            if ((isnan(stokesVectorTransmission.x) || isnan(stokesVectorTransmission.y) || isnan(stokesVectorTransmission.z) || isnan(stokesVectorTransmission.w)) && !(isnan(stokesVector.x) || isnan(stokesVector.y) || isnan(stokesVector.z) || isnan(stokesVector.w)))
            {
                printf("NaN in stokes vector after transmission with complex refractive indices\n");
                printf("cosTheta: %f, cosThetaPrime: %f, nA: (%f, %f), nB: (%f, %f)\n", cosTheta, cosThetaPrime, nA.real(), nA.imag(), nB.real(), nB.imag());
                printf("stokesVector before: (%f, %f, %f, %f)\n", stokesVector.x, stokesVector.y, stokesVector.z, stokesVector.w);
                printf("mullerMatrix: \n");
                for (int i = 0; i < 4; ++i)
                {
                    printf("(%f, %f, %f, %f)\n", mullerMatrix.m[i].x, mullerMatrix.m[i].y, mullerMatrix.m[i].z, mullerMatrix.m[i].w);
                }
                printf("stokesVectorTransmission: (%f, %f, %f, %f)\n", stokesVectorTransmission.x, stokesVectorTransmission.y, stokesVectorTransmission.z, stokesVectorTransmission.w);
            }
            stokesVector = stokesVectorTransmission;
            return false;
        }
    }
}

#endif
#endif