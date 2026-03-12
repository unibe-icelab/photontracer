#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/Transform4.h>
#include <algorithm>
#include <array>
#include <limits>

#include "../photonTracer/light_scattering.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Test snells law and fresnel equations
extern "C" __global__ void testAbsorption(
    float *travelled_distance, bool *absorbed, float *random_number, float *k)
{

    float max_distance = 1e-1f; // 100 microns
    float wavelengthUm = 0.55f; // 550 nm

    float lengthScale = 1e-3f;

    *absorbed = calculateAbsorption(max_distance, wavelengthUm, lengthScale, *k, *random_number, *travelled_distance);
}

TEST(LightScattering, Absorption)
{
    float res_travelled_distance;
    float travelled_distance;
    bool absorbed;
    float random_number;
    float k = 1.96e-9f;

    // Device pointers
    float *d_travelled_distance = nullptr;
    bool *d_absorbed = nullptr;
    float *d_random_number = nullptr;
    float *d_k = nullptr;

    // Allocate device memory
    cudaMalloc((void **)&d_travelled_distance, sizeof(float));
    cudaMalloc((void **)&d_absorbed, sizeof(bool));
    cudaMalloc((void **)&d_random_number, sizeof(float));
    cudaMalloc((void **)&d_k, sizeof(float));

    // Launch kernel
    random_number = 0.5f;
    res_travelled_distance = 1e-1f; // max_distance in mm
    cudaMemcpy(d_random_number, &random_number, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, &k, sizeof(float), cudaMemcpyHostToDevice);
    // run test kernel
    testAbsorption<<<1, 1>>>(d_travelled_distance, d_absorbed, d_random_number, d_k);
    // Copy data from device to host
    cudaMemcpy(&travelled_distance, d_travelled_distance, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&absorbed, d_absorbed, sizeof(bool), cudaMemcpyDeviceToHost);

    // Verify the result
    EXPECT_FALSE(absorbed);
    EXPECT_NEAR(travelled_distance, res_travelled_distance, 1e-4);

    random_number = 0.000001f;
    res_travelled_distance = 0.02233042; // 22um
    cudaMemcpy(d_random_number, &random_number, sizeof(float), cudaMemcpyHostToDevice);
    testAbsorption<<<1, 1>>>(d_travelled_distance, d_absorbed, d_random_number, d_k);

    cudaMemcpy(&travelled_distance, d_travelled_distance, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&absorbed, d_absorbed, sizeof(bool), cudaMemcpyDeviceToHost);

    EXPECT_TRUE(absorbed);
    EXPECT_NEAR(travelled_distance, res_travelled_distance, 1e-3);

    // Test on CPU
    random_number = 0.000001f;
    res_travelled_distance = 0.02233042; // 22um
    float max_distance = 1e-1f;          // 100 microns
    float wavelengthUm = 0.55f;          // 550 nm
    float lengthScale = 1e-3f;

    absorbed = calculateAbsorption(max_distance, wavelengthUm, lengthScale, k, random_number, travelled_distance);

    EXPECT_TRUE(absorbed);
    EXPECT_NEAR(travelled_distance, res_travelled_distance, 1e-3);

    k = 0.0f;
    random_number = 0.5f;
    cudaMemcpy(d_k, &k, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_random_number, &random_number, sizeof(float), cudaMemcpyHostToDevice);
    testAbsorption<<<1, 1>>>(d_travelled_distance, d_absorbed, d_random_number, d_k);
    cudaMemcpy(&absorbed, d_absorbed, sizeof(bool), cudaMemcpyDeviceToHost);

    EXPECT_FALSE(absorbed);

    // Free device memory
    cudaFree(d_travelled_distance);
    cudaFree(d_absorbed);
}

extern "C" __global__ void testSnellsLaw(
    float3 *incident_direction, float3 *normal, float *n1, float *n2,
    float *cos_theta, float *cos_theta_prime, bool *total_internal_reflection)
{
    calculateSnellsLaw(*incident_direction, *normal, *n1, *n2, *cos_theta, *cos_theta_prime, *total_internal_reflection);
}

TEST(LightScattering, SnellsLaw)
{
    float3 incident_direction, normal;
    float n1, n2, cos_theta, cos_theta_prime;
    bool total_internal_reflection;

    float res_cos_theta, res_cos_theta_prime;
    bool res_total_internal_reflection;

    constexpr float kCos50Deg = 0.64278758f;
    constexpr float kCos45Deg = 0.70710677f;
    constexpr float kCos67_8667Deg = 0.37676269f;
    constexpr float kCos32_6684Deg = 0.84180862f;

    // Device pointers
    float3 *d_incident_direction = nullptr;
    float3 *d_normal = nullptr;
    float *d_n1 = nullptr;
    float *d_n2 = nullptr;
    float *d_cos_theta = nullptr;
    float *d_cos_theta_prime = nullptr;
    bool *d_total_internal_reflection = nullptr;

    // Allocate device memory
    cudaMalloc((void **)&d_incident_direction, sizeof(float3));
    cudaMalloc((void **)&d_normal, sizeof(float3));
    cudaMalloc((void **)&d_n1, sizeof(float));
    cudaMalloc((void **)&d_n2, sizeof(float));
    cudaMalloc((void **)&d_cos_theta, sizeof(float));
    cudaMalloc((void **)&d_cos_theta_prime, sizeof(float));
    cudaMalloc((void **)&d_total_internal_reflection, sizeof(bool));

    // total internal reflection (ice -> air, 50 degrees)
    incident_direction = otk::normalize(make_float3(0.76604444311f, -0.64278760968f, 0.0f));
    normal = otk::normalize(make_float3(0.0f, 1.0f, 0.0f));
    n1 = 1.31f;
    n2 = 1.0f;

    cudaMemcpy(d_incident_direction, &incident_direction, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normal, &normal, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n1, &n1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n2, &n2, sizeof(float), cudaMemcpyHostToDevice);

    testSnellsLaw<<<1, 1>>>(d_incident_direction, d_normal, d_n1, d_n2, d_cos_theta, d_cos_theta_prime, d_total_internal_reflection);

    cudaMemcpy(&cos_theta, d_cos_theta, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_internal_reflection, d_total_internal_reflection, sizeof(bool), cudaMemcpyDeviceToHost);

    res_cos_theta = kCos50Deg;
    res_total_internal_reflection = true;

    EXPECT_NEAR(cos_theta, res_cos_theta, 1e-2);
    EXPECT_EQ(total_internal_reflection, res_total_internal_reflection);

    float host_cos_theta = 0.0f;
    float host_cos_theta_prime = 0.0f;
    bool host_total_internal_reflection = false;
    calculateSnellsLaw(incident_direction, normal, n1, n2, host_cos_theta, host_cos_theta_prime, host_total_internal_reflection);
    EXPECT_NEAR(host_cos_theta, res_cos_theta, 1e-4);
    EXPECT_EQ(host_total_internal_reflection, res_total_internal_reflection);
    EXPECT_FLOAT_EQ(host_cos_theta_prime, 0.0f);

    // refraction 1 (ice -> vacuum, 45 degrees -> 67.87 degrees)
    incident_direction = otk::normalize(make_float3(1.f, -1.f, 0.0f));
    normal = otk::normalize(make_float3(0.0f, 1.0f, 0.0f));
    n1 = 1.31f;
    n2 = 1.0f;

    cudaMemcpy(d_incident_direction, &incident_direction, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normal, &normal, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n1, &n1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n2, &n2, sizeof(float), cudaMemcpyHostToDevice);

    testSnellsLaw<<<1, 1>>>(d_incident_direction, d_normal, d_n1, d_n2, d_cos_theta, d_cos_theta_prime, d_total_internal_reflection);

    cudaMemcpy(&cos_theta, d_cos_theta, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&cos_theta_prime, d_cos_theta_prime, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_internal_reflection, d_total_internal_reflection, sizeof(bool), cudaMemcpyDeviceToHost);

    // control values
    res_cos_theta = kCos45Deg;
    res_cos_theta_prime = kCos67_8667Deg;
    res_total_internal_reflection = false;

    EXPECT_NEAR(cos_theta, res_cos_theta, 1e-2);
    EXPECT_NEAR(cos_theta_prime, res_cos_theta_prime, 1e-2);
    EXPECT_EQ(total_internal_reflection, res_total_internal_reflection);

    calculateSnellsLaw(incident_direction, normal, n1, n2, host_cos_theta, host_cos_theta_prime, host_total_internal_reflection);
    EXPECT_NEAR(host_cos_theta, res_cos_theta, 1e-4);
    EXPECT_NEAR(host_cos_theta_prime, res_cos_theta_prime, 1e-4);
    EXPECT_EQ(host_total_internal_reflection, res_total_internal_reflection);

    // refraction 1 (vacuum -> ice, -45 degrees -> 32.66 degrees)
    incident_direction = otk::normalize(make_float3(-1.f, -1.f, 0.0f));
    normal = otk::normalize(make_float3(0.0f, 1.0f, 0.0f));
    n1 = 1.0f;
    n2 = 1.31f;

    cudaMemcpy(d_incident_direction, &incident_direction, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normal, &normal, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n1, &n1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n2, &n2, sizeof(float), cudaMemcpyHostToDevice);

    testSnellsLaw<<<1, 1>>>(d_incident_direction, d_normal, d_n1, d_n2, d_cos_theta, d_cos_theta_prime, d_total_internal_reflection);

    cudaMemcpy(&cos_theta, d_cos_theta, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&cos_theta_prime, d_cos_theta_prime, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_internal_reflection, d_total_internal_reflection, sizeof(bool), cudaMemcpyDeviceToHost);

    // control values
    res_cos_theta = kCos45Deg;
    res_cos_theta_prime = kCos32_6684Deg;
    res_total_internal_reflection = false;

    EXPECT_NEAR(cos_theta, res_cos_theta, 1e-2);
    EXPECT_NEAR(cos_theta_prime, res_cos_theta_prime, 1e-2);
    EXPECT_EQ(total_internal_reflection, res_total_internal_reflection);

    calculateSnellsLaw(incident_direction, normal, n1, n2, host_cos_theta, host_cos_theta_prime, host_total_internal_reflection);
    EXPECT_NEAR(host_cos_theta, res_cos_theta, 1e-4);
    EXPECT_NEAR(host_cos_theta_prime, res_cos_theta_prime, 1e-4);
    EXPECT_EQ(host_total_internal_reflection, res_total_internal_reflection);

    // Free device memory
    cudaFree(d_incident_direction);
    cudaFree(d_normal);
    cudaFree(d_n1);
    cudaFree(d_n2);
    cudaFree(d_cos_theta);
    cudaFree(d_cos_theta_prime);
    cudaFree(d_total_internal_reflection);
}

extern "C" __global__ void testRefractedDirection(
    const float3 *direction, const float3 *normal,
    const float *cos_theta, const float *cos_theta_prime,
    const float *na, const float *nb,
    float3 *refracted_direction)
{
    *refracted_direction = calculateRefractedDirection(*direction, *normal, *cos_theta, *cos_theta_prime, *na, *nb);
}

TEST(LightScattering, RefractedDirection)
{
    float3 direction, normal, refracted_direction;
    float cos_theta, cos_theta_prime, na, nb;
    float3 res_refracted_direction;

    // Device pointers
    float3 *d_direction = nullptr;
    float3 *d_normal = nullptr;
    float3 *d_refracted_direction = nullptr;
    float *d_cos_theta = nullptr;
    float *d_cos_theta_prime = nullptr;
    float *d_na = nullptr;
    float *d_nb = nullptr;

    // Allocate device memory
    cudaMalloc((void **)&d_direction, sizeof(float3));
    cudaMalloc((void **)&d_normal, sizeof(float3));
    cudaMalloc((void **)&d_refracted_direction, sizeof(float3));
    cudaMalloc((void **)&d_cos_theta, sizeof(float));
    cudaMalloc((void **)&d_cos_theta_prime, sizeof(float));
    cudaMalloc((void **)&d_na, sizeof(float));
    cudaMalloc((void **)&d_nb, sizeof(float));

    // refracted direction (vacuum -> ice, 45 degrees)
    direction = otk::normalize(make_float3(1, -1, 0.0f));
    normal = otk::normalize(make_float3(0.0f, 1.0f, 0.0f));
    cos_theta = cosf(45.0f * M_PI / 180.0f);
    cos_theta_prime = cosf(32.3905 * M_PI / 180.0f);
    na = 1.0f;
    nb = 1.32f;

    res_refracted_direction = make_float3(0.5357f, -0.8444f, 0.0f);

    cudaMemcpy(d_direction, &direction, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normal, &normal, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos_theta, &cos_theta, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos_theta_prime, &cos_theta_prime, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_na, &na, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb, &nb, sizeof(float), cudaMemcpyHostToDevice);

    testRefractedDirection<<<1, 1>>>(d_direction, d_normal, d_cos_theta, d_cos_theta_prime, d_na, d_nb, d_refracted_direction);

    cudaMemcpy(&refracted_direction, d_refracted_direction, sizeof(float3), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(refracted_direction.x, res_refracted_direction.x, 1e-4);
    EXPECT_NEAR(refracted_direction.y, res_refracted_direction.y, 1e-4);
    EXPECT_NEAR(refracted_direction.z, res_refracted_direction.z, 1e-4);

    // refracted direction (ice -> vacuum, 45 degrees)
    direction = otk::normalize(make_float3(0.530861f, 0.000000f, -0.847459f));
    cos_theta = cosf(45.0f * M_PI / 180.0f);
    cos_theta_prime = cosf(68.968 * M_PI / 180.0f);
    na = 1.32f;
    nb = 1.0f;

    res_refracted_direction = make_float3(0.93338f, -0.35889f, 0.0f);

    cudaMemcpy(d_cos_theta, &cos_theta, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos_theta_prime, &cos_theta_prime, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_na, &na, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb, &nb, sizeof(float), cudaMemcpyHostToDevice);

    testRefractedDirection<<<1, 1>>>(d_direction, d_normal, d_cos_theta, d_cos_theta_prime, d_na, d_nb, d_refracted_direction);

    cudaMemcpy(&refracted_direction, d_refracted_direction, sizeof(float3), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(refracted_direction.x, res_refracted_direction.x, 1e-4);
    EXPECT_NEAR(refracted_direction.y, res_refracted_direction.y, 1e-4);
    EXPECT_NEAR(refracted_direction.z, res_refracted_direction.z, 1e-4);

    // refracted direction (ice -> vacuum, 57 degrees)
    cos_theta = 0.847459;
    cos_theta_prime = 0.707107;
    na = 1.332f;
    nb = 1.0f;

    direction = make_float3(0.530861f, 0.000000f, 0.847459f);
    normal = make_float3(0.0f, 0.0f, -1.0f);
    res_refracted_direction = make_float3(0.707107f, 0.0f, 0.707107f);

    cudaMemcpy(d_cos_theta, &cos_theta, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos_theta_prime, &cos_theta_prime, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_na, &na, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb, &nb, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_direction, &direction, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normal, &normal, sizeof(float3), cudaMemcpyHostToDevice);

    testRefractedDirection<<<1, 1>>>(d_direction, d_normal, d_cos_theta, d_cos_theta_prime, d_na, d_nb, d_refracted_direction);

    cudaMemcpy(&refracted_direction, d_refracted_direction, sizeof(float3), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(refracted_direction.x, res_refracted_direction.x, 1e-4);
    EXPECT_NEAR(refracted_direction.y, res_refracted_direction.y, 1e-4);
    EXPECT_NEAR(refracted_direction.z, res_refracted_direction.z, 1e-4);

    // Free device memory
    cudaFree(d_direction);
    cudaFree(d_normal);
    cudaFree(d_refracted_direction);
    cudaFree(d_cos_theta);
    cudaFree(d_cos_theta_prime);
    cudaFree(d_na);
    cudaFree(d_nb);
}

extern "C" __global__ void testReflectedDirection(const float3 *direction, const float3 *normal, float3 *reflected_ray)
{
    *reflected_ray = calculateReflectedDirection(*direction, *normal);
}

TEST(LightScattering, ReflectedDirection)
{
    float3 direction, normal, reflected_ray;
    float3 res_reflected_ray;

    // Device pointers
    float3 *d_direction = nullptr;
    float3 *d_normal = nullptr;
    float3 *d_reflected_ray = nullptr;

    // Allocate device memory
    cudaMalloc((void **)&d_direction, sizeof(float3));
    cudaMalloc((void **)&d_normal, sizeof(float3));
    cudaMalloc((void **)&d_reflected_ray, sizeof(float3));

    // reflected direction (45 degrees)
    direction = otk::normalize(make_float3(1, -1, 0.0f));
    normal = otk::normalize(make_float3(0.0f, 1.0f, 0.0f));

    res_reflected_ray = otk::normalize(make_float3(1.0f, 1.0f, 0.0f));

    cudaMemcpy(d_direction, &direction, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normal, &normal, sizeof(float3), cudaMemcpyHostToDevice);

    testReflectedDirection<<<1, 1>>>(d_direction, d_normal, d_reflected_ray);

    cudaMemcpy(&reflected_ray, d_reflected_ray, sizeof(float3), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(reflected_ray.x, res_reflected_ray.x, 1e-4);
    EXPECT_NEAR(reflected_ray.y, res_reflected_ray.y, 1e-4);
    EXPECT_NEAR(reflected_ray.z, res_reflected_ray.z, 1e-4);

    // reflected direction (45 degrees)
    direction = otk::normalize(make_float3(100.0f, -1.0f, 100.0f));
    normal = otk::normalize(make_float3(0.0f, 1.0f, 0.0f));

    res_reflected_ray = otk::normalize(make_float3(100.0f, 1.0f, 100.0f));

    cudaMemcpy(d_direction, &direction, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normal, &normal, sizeof(float3), cudaMemcpyHostToDevice);

    testReflectedDirection<<<1, 1>>>(d_direction, d_normal, d_reflected_ray);

    cudaMemcpy(&reflected_ray, d_reflected_ray, sizeof(float3), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(reflected_ray.x, res_reflected_ray.x, 1e-4);
    EXPECT_NEAR(reflected_ray.y, res_reflected_ray.y, 1e-4);
    EXPECT_NEAR(reflected_ray.z, res_reflected_ray.z, 1e-4);
}

TEST(LightScattering, SignedRotationAboutAxis)
{
    const float3 axis = otk::normalize(make_float3(0.0f, 0.0f, 1.0f));
    const float3 vector1 = otk::normalize(make_float3(1.0f, 0.0f, 0.0f));

    float3 vector2 = otk::normalize(make_float3(0.0f, 1.0f, 0.0f));
    float angle = signedRotationAboutAxis(axis, vector1, vector2);
    EXPECT_NEAR(angle, static_cast<float>(M_PI / 2.0), 1e-5);

    vector2 = otk::normalize(make_float3(0.0f, -1.0f, 0.0f));
    angle = signedRotationAboutAxis(axis, vector1, vector2);
    EXPECT_NEAR(angle, static_cast<float>(-M_PI / 2.0), 1e-5);

    const float3 flippedAxis = -axis;
    angle = signedRotationAboutAxis(flippedAxis, vector1, otk::normalize(make_float3(0.0f, 1.0f, 0.0f)));
    EXPECT_NEAR(angle, static_cast<float>(-M_PI / 2.0), 1e-5);
}

TEST(LightScattering, ScatteringPlaneAxes)
{
    const float3 k = otk::normalize(make_float3(0.2f, -1.0f, 0.3f));
    const float3 surfaceNormal = otk::normalize(make_float3(0.0f, 1.0f, 0.0f));
    float3 scatteringPlaneNormal;

    scatteringPlaneNormalAxis(k, surfaceNormal, scatteringPlaneNormal);

    const float3 expected = otk::normalize(otk::cross(surfaceNormal, k));
    EXPECT_NEAR(scatteringPlaneNormal.x, expected.x, 1e-6);
    EXPECT_NEAR(scatteringPlaneNormal.y, expected.y, 1e-6);
    EXPECT_NEAR(scatteringPlaneNormal.z, expected.z, 1e-6);

    EXPECT_NEAR(otk::dot(scatteringPlaneNormal, k), 0.0f, 1e-5);
    EXPECT_NEAR(otk::dot(scatteringPlaneNormal, surfaceNormal), 0.0f, 1e-5);
}

extern "C" __global__ void testLambertianDirectionKernel(float randomNumber1, float randomNumber2, float3 normal, float3 *direction)
{
    *direction = calculateLamberianDirection(randomNumber1, randomNumber2, normal);
}

TEST(LightScattering, LambertianDirection)
{
    const float randomNumber1 = 0.25f;
    const float randomNumber2 = 0.75f;
    const float3 normal = otk::normalize(make_float3(0.0f, 1.0f, 0.0f));

    float3 *d_direction = nullptr;
    cudaMalloc((void **)&d_direction, sizeof(float3));

    testLambertianDirectionKernel<<<1, 1>>>(randomNumber1, randomNumber2, normal, d_direction);

    float3 direction;
    cudaMemcpy(&direction, d_direction, sizeof(float3), cudaMemcpyDeviceToHost);

    const float theta = 2.0f * static_cast<float>(M_PI) * randomNumber1;
    const float phi = acosf(2.0f * randomNumber2 - 1.0f);
    const float x = sinf(phi) * cosf(theta);
    const float y = sinf(phi) * sinf(theta);
    const float z = cosf(phi);
    const float3 expected = otk::normalize(normal + make_float3(x, y, z));

    EXPECT_NEAR(direction.x, expected.x, 1e-5);
    EXPECT_NEAR(direction.y, expected.y, 1e-5);
    EXPECT_NEAR(direction.z, expected.z, 1e-5);

    EXPECT_NEAR(otk::length(direction), 1.0f, 1e-5);
    EXPECT_GT(otk::dot(direction, normal), 0.0f);

    cudaFree(d_direction);
}

extern "C" __global__ void testReflectivitySchlickKernel(float *result, float cosTheta, float na, float nb)
{
    *result = calculateReflectivitySchlick(cosTheta, na, nb);
}

extern "C" __global__ void testMullerTotalReflectionKernel(float cosThetaA, float na, float nb, otk::Transform4 *matrix)
{
    *matrix = calculateMullerTotalReflection(cosThetaA, na, nb);
}

extern "C" __global__ void testMullerReflectionKernel(float cosThetaA, float cosThetaB, float na, float nb, otk::Transform4 *matrix)
{
    *matrix = calculateMullerReflection(cosThetaA, cosThetaB, na, nb);
}

extern "C" __global__ void testMullerTransmissionKernel(float cosThetaA, float cosThetaB, float na, float nb, otk::Transform4 *matrix)
{
    *matrix = calculateMullerTransmission(cosThetaA, cosThetaB, na, nb);
}

TEST(LightScattering, ReflectivitySchlick)
{
    float *d_result = nullptr;
    cudaMalloc((void **)&d_result, sizeof(float));

    const float na = 1.33f;
    const float nb = 1.0f;
    float cosTheta = 1.0f;

    testReflectivitySchlickKernel<<<1, 1>>>(d_result, cosTheta, na, nb);

    float reflectivity = 0.0f;
    cudaMemcpy(&reflectivity, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    const float r0 = powf((na - nb) / (na + nb), 2.0f);
    EXPECT_NEAR(reflectivity, r0, 1e-6);

    cosTheta = 0.25f;
    testReflectivitySchlickKernel<<<1, 1>>>(d_result, cosTheta, na, nb);
    cudaMemcpy(&reflectivity, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    const float expected = r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);
    EXPECT_NEAR(reflectivity, expected, 1e-6);

    cudaFree(d_result);
}

TEST(LightScattering, MullerTotalReflection)
{
    const float cosThetaA = cosf(60.0f * static_cast<float>(M_PI) / 180.0f);
    const float na = 1.5f;
    const float nb = 1.0f;

    const otk::Transform4 hostMatrix = calculateMullerTotalReflection(cosThetaA, na, nb);

    otk::Transform4 *d_matrix = nullptr;
    cudaMalloc((void **)&d_matrix, sizeof(otk::Transform4));
    testMullerTotalReflectionKernel<<<1, 1>>>(cosThetaA, na, nb, d_matrix);

    otk::Transform4 deviceMatrix{};
    cudaMemcpy(&deviceMatrix, d_matrix, sizeof(otk::Transform4), cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);

    auto verifyMatrix = [](const otk::Transform4 &matrix)
    {
        EXPECT_NEAR(matrix.m[0].x, 1.0f, 1e-6);
        EXPECT_NEAR(matrix.m[1].y, 1.0f, 1e-6);
        EXPECT_NEAR(matrix.m[1].x, 0.0f, 1e-6);

        const float expectedCos = 0.76086956f;
        const float expectedSin = -0.64890486f;

        EXPECT_NEAR(matrix.m[2].z, expectedCos, 1e-5);
        EXPECT_NEAR(matrix.m[2].w, expectedSin, 1e-5);
        EXPECT_NEAR(matrix.m[3].z, -expectedSin, 1e-5);
        EXPECT_NEAR(matrix.m[3].w, expectedCos, 1e-5);

        const float rotationNorm = matrix.m[2].z * matrix.m[2].z + matrix.m[2].w * matrix.m[2].w;
        EXPECT_NEAR(rotationNorm, 1.0f, 1e-5);
    };

    verifyMatrix(hostMatrix);
    verifyMatrix(deviceMatrix);

    for (int row = 0; row < 4; ++row)
    {
        EXPECT_NEAR(deviceMatrix.m[row].x, hostMatrix.m[row].x, 1e-6);
        EXPECT_NEAR(deviceMatrix.m[row].y, hostMatrix.m[row].y, 1e-6);
        EXPECT_NEAR(deviceMatrix.m[row].z, hostMatrix.m[row].z, 1e-6);
        EXPECT_NEAR(deviceMatrix.m[row].w, hostMatrix.m[row].w, 1e-6);
    }
}

TEST(LightScattering, MullerReflection)
{
    const float cosThetaA = cosf(45.0f * static_cast<float>(M_PI) / 180.0f);
    const float cosThetaB = cosf(28.125f * static_cast<float>(M_PI) / 180.0f);
    const float na = 1.0f;
    const float nb = 1.33f;

    const otk::Transform4 hostMatrix = calculateMullerReflection(cosThetaA, cosThetaB, na, nb);

    otk::Transform4 *d_matrix = nullptr;
    cudaMalloc((void **)&d_matrix, sizeof(otk::Transform4));
    testMullerReflectionKernel<<<1, 1>>>(cosThetaA, cosThetaB, na, nb, d_matrix);

    otk::Transform4 deviceMatrix{};
    cudaMemcpy(&deviceMatrix, d_matrix, sizeof(otk::Transform4), cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);

    auto verifyMatrix = [](const otk::Transform4 &matrix)
    {
        const float rSrPP = 0.031214129f;
        const float rSrPM = -0.030182571f;
        const float rSrP = -0.0079582815f;

        EXPECT_NEAR(matrix.m[0].x, rSrPP, 1e-6);
        EXPECT_NEAR(matrix.m[0].y, rSrPM, 1e-6);
        EXPECT_NEAR(matrix.m[1].x, rSrPM, 1e-6);
        EXPECT_NEAR(matrix.m[1].y, rSrPP, 1e-6);
        EXPECT_NEAR(matrix.m[2].z, rSrP, 1e-6);
        EXPECT_NEAR(matrix.m[3].w, rSrP, 1e-6);
    };

    verifyMatrix(hostMatrix);
    verifyMatrix(deviceMatrix);

    for (int row = 0; row < 4; ++row)
    {
        EXPECT_NEAR(deviceMatrix.m[row].x, hostMatrix.m[row].x, 1e-6);
        EXPECT_NEAR(deviceMatrix.m[row].y, hostMatrix.m[row].y, 1e-6);
        EXPECT_NEAR(deviceMatrix.m[row].z, hostMatrix.m[row].z, 1e-6);
        EXPECT_NEAR(deviceMatrix.m[row].w, hostMatrix.m[row].w, 1e-6);
    }
}

TEST(LightScattering, MullerTransmission)
{
    const float cosThetaA = 1.0f;
    const float cosThetaB = 1.0f;
    const float na = 1.0f;
    const float nb = 1.0f;

    const otk::Transform4 hostMatrix = calculateMullerTransmission(cosThetaA, cosThetaB, na, nb);

    otk::Transform4 *d_matrix = nullptr;
    cudaMalloc((void **)&d_matrix, sizeof(otk::Transform4));
    testMullerTransmissionKernel<<<1, 1>>>(cosThetaA, cosThetaB, na, nb, d_matrix);

    otk::Transform4 deviceMatrix{};
    cudaMemcpy(&deviceMatrix, d_matrix, sizeof(otk::Transform4), cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);

    auto verifyMatrix = [](const otk::Transform4 &matrix)
    {
        const otk::Transform4 identity = otk::identity();
        for (int i = 0; i < 4; ++i)
        {
            EXPECT_NEAR(matrix.m[i].x, identity.m[i].x, 1e-6);
            EXPECT_NEAR(matrix.m[i].y, identity.m[i].y, 1e-6);
            EXPECT_NEAR(matrix.m[i].z, identity.m[i].z, 1e-6);
            EXPECT_NEAR(matrix.m[i].w, identity.m[i].w, 1e-6);
        }
    };

    verifyMatrix(hostMatrix);
    verifyMatrix(deviceMatrix);

    for (int row = 0; row < 4; ++row)
    {
        EXPECT_NEAR(deviceMatrix.m[row].x, hostMatrix.m[row].x, 1e-6);
        EXPECT_NEAR(deviceMatrix.m[row].y, hostMatrix.m[row].y, 1e-6);
        EXPECT_NEAR(deviceMatrix.m[row].z, hostMatrix.m[row].z, 1e-6);
        EXPECT_NEAR(deviceMatrix.m[row].w, hostMatrix.m[row].w, 1e-6);
    }
}

extern "C" __global__ void testFresnelInteractionKernel(
    float4 *stokesVector,
    float3 *qMinusAxis,
    float3 *direction,
    const float3 *surfaceNormal,
    float nA,
    float nB,
    float randomSample,
    bool *reflected)
{
    float4 stokes = *stokesVector;
    float3 qAxis = *qMinusAxis;
    float3 dir = *direction;
    const bool isReflected = calculateFresnelInteraction(stokes, qAxis, dir, *surfaceNormal, nA, nB, randomSample);
    *stokesVector = stokes;
    *qMinusAxis = qAxis;
    *direction = dir;
    *reflected = isReflected;
}

TEST(LightScattering, FresnelInteraction)
{
    float4 *d_stokes = nullptr;
    float3 *d_qAxis = nullptr;
    float3 *d_direction = nullptr;
    float3 *d_surfaceNormal = nullptr;
    bool *d_reflected = nullptr;

    cudaMalloc((void **)&d_stokes, sizeof(float4));
    cudaMalloc((void **)&d_qAxis, sizeof(float3));
    cudaMalloc((void **)&d_direction, sizeof(float3));
    cudaMalloc((void **)&d_surfaceNormal, sizeof(float3));
    cudaMalloc((void **)&d_reflected, sizeof(bool));

    const float3 surfaceNormal = otk::normalize(make_float3(0.0f, 1.0f, 0.0f));
    cudaMemcpy(d_surfaceNormal, &surfaceNormal, sizeof(float3), cudaMemcpyHostToDevice);

    const float3 incidentDirection = otk::normalize(make_float3(1.0f, -1.0f, 0.0f)); // 45° incidence
    float3 scatteringAxis;
    scatteringPlaneNormalAxis(incidentDirection, surfaceNormal, scatteringAxis);

    const float nAir = 1.0f;
    const float nIce = 1.33f;

    float cosTheta = 0.0f;
    float cosThetaPrime = 0.0f;
    bool tir = false;
    calculateSnellsLaw(incidentDirection, surfaceNormal, nAir, nIce, cosTheta, cosThetaPrime, tir);
    ASSERT_FALSE(tir);

    const float3 expectedReflectionDirection = calculateReflectedDirection(incidentDirection, surfaceNormal);
    const float3 expectedTransmissionDirection = calculateRefractedDirection(incidentDirection, surfaceNormal, cosTheta, cosThetaPrime, nAir, nIce);
    const otk::Transform4 reflectionMatrix = calculateMullerReflection(cosTheta, cosThetaPrime, nAir, nIce);
    const otk::Transform4 transmissionMatrix = calculateMullerTransmission(cosTheta, cosThetaPrime, nAir, nIce);

    struct FresnelResult
    {
        float4 stokes;
        float3 qAxis;
        float3 direction;
        bool reflected;
    };

    auto runInteraction = [&](const float4 &initialStokes, float randomSample) -> FresnelResult
    {
        FresnelResult result{};
        // Use a finite initial Q- axis, matching the simulation behavior.
        float3 initialAxis = scatteringAxis;
        float3 direction = incidentDirection;

        cudaMemcpy(d_stokes, &initialStokes, sizeof(float4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_qAxis, &initialAxis, sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_direction, &direction, sizeof(float3), cudaMemcpyHostToDevice);

        testFresnelInteractionKernel<<<1, 1>>>(d_stokes, d_qAxis, d_direction, d_surfaceNormal, nAir, nIce, randomSample, d_reflected);

        cudaMemcpy(&result.stokes, d_stokes, sizeof(float4), cudaMemcpyDeviceToHost);
        cudaMemcpy(&result.qAxis, d_qAxis, sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(&result.direction, d_direction, sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(&result.reflected, d_reflected, sizeof(bool), cudaMemcpyDeviceToHost);
        return result;
    };

    const float4 stokesPlusQ = make_float4(1.0f, 1.0f, 0.0f, 0.0f);
    const float4 stokesMinusQ = make_float4(1.0f, -1.0f, 0.0f, 0.0f);
    const float4 stokesUnpolarized = make_float4(1.0f, 0.0f, 0.0f, 0.0f);

    const float transmissivityPlus = (transmissionMatrix * stokesPlusQ).x;
    ASSERT_LT(transmissivityPlus, 0.999f);

    const FresnelResult reflectionPlusQ = runInteraction(stokesPlusQ, 0.999f);
    EXPECT_TRUE(reflectionPlusQ.reflected);
    EXPECT_NEAR(reflectionPlusQ.direction.x, expectedReflectionDirection.x, 1e-5);
    EXPECT_NEAR(reflectionPlusQ.direction.y, expectedReflectionDirection.y, 1e-5);
    EXPECT_NEAR(reflectionPlusQ.direction.z, expectedReflectionDirection.z, 1e-5);
    EXPECT_NEAR(reflectionPlusQ.qAxis.x, scatteringAxis.x, 1e-5);
    EXPECT_NEAR(reflectionPlusQ.qAxis.y, scatteringAxis.y, 1e-5);
    EXPECT_NEAR(reflectionPlusQ.qAxis.z, scatteringAxis.z, 1e-5);

    float4 expectedReflectedStokes = reflectionMatrix * stokesPlusQ;
    expectedReflectedStokes = expectedReflectedStokes / expectedReflectedStokes.x;
    EXPECT_NEAR(reflectionPlusQ.stokes.x, expectedReflectedStokes.x, 1e-5);
    EXPECT_NEAR(reflectionPlusQ.stokes.y, expectedReflectedStokes.y, 1e-5);
    EXPECT_NEAR(reflectionPlusQ.stokes.z, expectedReflectedStokes.z, 1e-5);
    EXPECT_NEAR(reflectionPlusQ.stokes.w, expectedReflectedStokes.w, 1e-5);

    const FresnelResult transmissionMinusQ = runInteraction(stokesMinusQ, 0.0f);
    EXPECT_FALSE(transmissionMinusQ.reflected);
    EXPECT_NEAR(transmissionMinusQ.direction.x, expectedTransmissionDirection.x, 1e-5);
    EXPECT_NEAR(transmissionMinusQ.direction.y, expectedTransmissionDirection.y, 1e-5);
    EXPECT_NEAR(transmissionMinusQ.direction.z, expectedTransmissionDirection.z, 1e-5);
    EXPECT_NEAR(transmissionMinusQ.qAxis.x, scatteringAxis.x, 1e-5);
    EXPECT_NEAR(transmissionMinusQ.qAxis.y, scatteringAxis.y, 1e-5);
    EXPECT_NEAR(transmissionMinusQ.qAxis.z, scatteringAxis.z, 1e-5);

    float4 expectedTransmissionMinusQ = transmissionMatrix * stokesMinusQ;
    const float transmissivityMinus = expectedTransmissionMinusQ.x;
    ASSERT_GT(transmissivityMinus, 0.0f);
    expectedTransmissionMinusQ = expectedTransmissionMinusQ / transmissivityMinus;
    EXPECT_NEAR(transmissionMinusQ.stokes.x, expectedTransmissionMinusQ.x, 1e-5);
    EXPECT_NEAR(transmissionMinusQ.stokes.y, expectedTransmissionMinusQ.y, 1e-5);
    EXPECT_NEAR(transmissionMinusQ.stokes.z, expectedTransmissionMinusQ.z, 1e-5);
    EXPECT_NEAR(transmissionMinusQ.stokes.w, expectedTransmissionMinusQ.w, 1e-5);

    const FresnelResult transmissionUnpolarized = runInteraction(stokesUnpolarized, 0.0f);
    EXPECT_FALSE(transmissionUnpolarized.reflected);
    EXPECT_NEAR(transmissionUnpolarized.direction.x, expectedTransmissionDirection.x, 1e-5);
    EXPECT_NEAR(transmissionUnpolarized.direction.y, expectedTransmissionDirection.y, 1e-5);
    EXPECT_NEAR(transmissionUnpolarized.direction.z, expectedTransmissionDirection.z, 1e-5);
    EXPECT_NEAR(transmissionUnpolarized.qAxis.x, scatteringAxis.x, 1e-5);
    EXPECT_NEAR(transmissionUnpolarized.qAxis.y, scatteringAxis.y, 1e-5);
    EXPECT_NEAR(transmissionUnpolarized.qAxis.z, scatteringAxis.z, 1e-5);

    float4 expectedTransmissionUnpolarized = transmissionMatrix * stokesUnpolarized;
    const float transmissivityUnpolarized = expectedTransmissionUnpolarized.x;
    ASSERT_GT(transmissivityUnpolarized, 0.0f);
    expectedTransmissionUnpolarized = expectedTransmissionUnpolarized / transmissivityUnpolarized;
    EXPECT_NEAR(transmissionUnpolarized.stokes.x, expectedTransmissionUnpolarized.x, 1e-5);
    EXPECT_NEAR(transmissionUnpolarized.stokes.y, expectedTransmissionUnpolarized.y, 1e-5);
    EXPECT_NEAR(transmissionUnpolarized.stokes.z, expectedTransmissionUnpolarized.z, 1e-5);
    EXPECT_NEAR(transmissionUnpolarized.stokes.w, expectedTransmissionUnpolarized.w, 1e-5);

    cudaFree(d_stokes);
    cudaFree(d_qAxis);
    cudaFree(d_direction);
    cudaFree(d_surfaceNormal);
    cudaFree(d_reflected);
}

extern "C" __global__ void testRotationMatrix(const float *angle, otk::Transform4 *rotation_matrix)
{
    *rotation_matrix = calculateRotationMatrix(*angle);
}

TEST(LightScattering, RotationMatrix)
{
    float omega;
    otk::Transform4 rotation_matrix;
    otk::Transform4 res_rotation_matrix;

    // Device pointers
    float *d_omega = nullptr;
    otk::Transform4 *d_rotation_matrix = nullptr;

    // Allocate device memory
    cudaMalloc((void **)&d_omega, sizeof(float));
    cudaMalloc((void **)&d_rotation_matrix, sizeof(otk::Transform4));

    omega = 30.0f * M_PI / 180.0f;

    float cos_2_omega = 0.5f;
    float sin_2_omega = 0.86603f;

    res_rotation_matrix.m[0] = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    res_rotation_matrix.m[1] = make_float4(0.0f, cos_2_omega, sin_2_omega, 0.0f);
    res_rotation_matrix.m[2] = make_float4(0.0f, -sin_2_omega, cos_2_omega, 0.0f);
    res_rotation_matrix.m[3] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    cudaMemcpy(d_omega, &omega, sizeof(float), cudaMemcpyHostToDevice);

    testRotationMatrix<<<1, 1>>>(d_omega, d_rotation_matrix);

    cudaMemcpy(&rotation_matrix, d_rotation_matrix, sizeof(otk::Transform4), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++)
    {
        EXPECT_NEAR(rotation_matrix.m[i].x, res_rotation_matrix.m[i].x, 1e-4);
        EXPECT_NEAR(rotation_matrix.m[i].y, res_rotation_matrix.m[i].y, 1e-4);
        EXPECT_NEAR(rotation_matrix.m[i].z, res_rotation_matrix.m[i].z, 1e-4);
        EXPECT_NEAR(rotation_matrix.m[i].w, res_rotation_matrix.m[i].w, 1e-4);
    }
}

TEST(MediumHistory, AppendPacked)
{
    uint32_t packed = 0;
    uint32_t size = 0;

    for (uint32_t medium = 1; medium <= 8; ++medium)
    {
        EXPECT_TRUE(appendMediumPacked(medium, size, packed));
        EXPECT_EQ(size, medium);
    }

    EXPECT_EQ(size, 8u);
    uint32_t history[8] = {};
    unpackMediumHistory(packed, history);
    for (uint32_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(history[i], i + 1);
    }

    // Overflow should fail and keep state intact
    EXPECT_FALSE(appendMediumPacked(9u, size, packed));
    EXPECT_EQ(size, 8u);
    unpackMediumHistory(packed, history);
    for (uint32_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(history[i], i + 1);
    }
}

TEST(MediumHistory, RemovePacked)
{
    uint32_t packed = 0;
    uint32_t size = 0;
    const std::array<uint32_t, 5> mediums{1, 2, 15, 2, 4};

    for (uint32_t medium : mediums)
    {
        ASSERT_TRUE(appendMediumPacked(medium, size, packed));
    }
    ASSERT_EQ(size, mediums.size());

    EXPECT_TRUE(removeLastOccurencePacked(2u, size, packed));
    EXPECT_EQ(size, mediums.size() - 1);
    uint32_t history[8] = {};
    unpackMediumHistory(packed, history);
    EXPECT_EQ(history[0], 1u);
    EXPECT_EQ(history[1], 2u);
    EXPECT_EQ(history[2], 15u);
    EXPECT_EQ(history[3], 4u);

    EXPECT_TRUE(removeLastOccurencePacked(1u, size, packed));
    EXPECT_EQ(size, mediums.size() - 2);
    unpackMediumHistory(packed, history);
    EXPECT_EQ(history[0], 2u);
    EXPECT_EQ(history[1], 15u);
    EXPECT_EQ(history[2], 4u);

    EXPECT_FALSE(removeLastOccurencePacked(7u, size, packed));
    EXPECT_EQ(size, mediums.size() - 2);

    // Clearing all entries should zero the packed buffer
    EXPECT_TRUE(removeLastOccurencePacked(4u, size, packed));
    EXPECT_TRUE(removeLastOccurencePacked(15u, size, packed));
    EXPECT_TRUE(removeLastOccurencePacked(2u, size, packed));
    EXPECT_EQ(size, 0u);
    EXPECT_EQ(packed, 0u);
}

TEST(MediumHistory, GetFromPacked)
{
    uint32_t history[8] = {1, 15, 3, 0, 9, 1, 12, 7};
    const uint32_t packed = packMediumHistory(history);

    // Out-of-range lookups should safely return zero without touching data
    EXPECT_EQ(getMediumFromPacked(packed, 8), 0);

    for (uint32_t i = 0; i < 8; ++i)
    {
        EXPECT_EQ(getMediumFromPacked(packed, i), history[i]);
    }
}

TEST(MediumHistory, AppendRemoveGetCombo)
{
    uint32_t packed = 0;
    uint32_t size = 0;

    const std::array<uint32_t, 6> initial{4, 7, 4, 9, 2, 9};
    for (uint32_t medium : initial)
    {
        ASSERT_TRUE(appendMediumPacked(medium, size, packed));
    }
    ASSERT_EQ(size, initial.size());

    // Verify the tail before removals
    EXPECT_EQ(getMediumFromPacked(packed, size - 1), 9u);
    EXPECT_EQ(getMediumFromPacked(packed, size - 2), 2u);

    // Remove a repeated medium and ensure the last occurrence vanished
    EXPECT_TRUE(removeLastOccurencePacked(4u, size, packed));
    EXPECT_EQ(size, initial.size() - 1);
    EXPECT_EQ(getMediumFromPacked(packed, 0), 4u);
    EXPECT_EQ(getMediumFromPacked(packed, 1), 7u);
    EXPECT_EQ(getMediumFromPacked(packed, 2), 9u);
    EXPECT_EQ(getMediumFromPacked(packed, 3), 2u);
    EXPECT_EQ(getMediumFromPacked(packed, 4), 9u);

    // Append again and confirm the new data is retrievable at the end
    ASSERT_TRUE(appendMediumPacked(5u, size, packed));
    EXPECT_EQ(getMediumFromPacked(packed, size - 1), 5u);

    // Removing a non-existent medium leaves state unchanged
    EXPECT_FALSE(removeLastOccurencePacked(11u, size, packed));
    EXPECT_EQ(getMediumFromPacked(packed, size - 1), 5u);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}