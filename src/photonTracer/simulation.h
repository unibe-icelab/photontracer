// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

// Portions of this code were derived from NVIDIA OptiX sample code.
// See THIRD_PARTY_NOTICES.md for full license text.

#pragma once

#include <optix.h>
#include <memory>
#include "i_geometry.h"
#include "ray_generator.h"
#include "raytracing_pipeline.h"
#include "raytracing_output.h"
#include "output_buffers.h"

class Simulation
{
public:
    Simulation(int gpuId = 0, int optixLoggingLevel = 1, bool enableValidationMode = false);
    ~Simulation();

    /**
     * Callback function for OptiX logging.
     * @param level The logging level.
     * @param tag The tag for the log message.
     * @param message The log message.
     * @param cbdata User data passed to the callback.
     */
    static void contextLogCb(uint32_t level, const char *tag,
                             const char *message, void * /*cbdata*/);

    /**
     * Initialize CUDA, select GPU, and create an OptiX device context.
     * @param gpuId The ID of the GPU to use (0-based).
     * @param optixLoggingLevel The logging level for OptiX (0-4).
     * @param enableValidationMode Enable OptiX validation mode if true.
     * @returns 0 on success, non-zero on failure.
     */
    void initializeContext(int gpuId, int optixLoggingLevel = 1, bool enableValidationMode = false);

    /**
     * Get the OptiX device context.
     * @returns The OptiX device context.
     */
    OptixDeviceContext getContext() const { return context_; }

    /**
     * Build the geometry for the simulation. This is done recursively for nested geometries.
     * @param geometry Pointer to the geometry object.
     */
    void buildGeometry(std::shared_ptr<IGeometry> geometry);

    /**
     * Set the geometry for the simulation.
     * @param geometry Pointer to the geometry object.
     */
    void setGeometry(std::shared_ptr<IGeometry> geometry);

    /**
     * Get a pointer to the geometry used in the simulation.
     * @returns Pointer to the geometry object.
     */
    std::shared_ptr<IGeometry> getGeometry() const;

    /**
     * Set the materials for the simulation.
     * @param materials Vector of materials to be used in the simulation.
     */
    void setMaterials(const std::vector<Material> &materials);

    /**
     * Get the materials used in the simulation.
     * @returns Vector of materials.
     */
    std::vector<Material> getMaterials() const;

    /**
     * Get the materials used in the simulation.
     * @returns Vector of materials.
     */
    void setWavelengthUm(float wavelengthUm);

    /**
     * Get the wavelength in micrometers used in the simulation.
     * @returns Wavelength in micrometers.
     */
    float getWavelengthUm() const;

    /**
     * Set the length unit of the simulation.
     * @param lengthUnit The length unit to be used in the simulation.
     */
    void setLengthUnit(LengthUnit lengthUnit);
    /**
     * Get the length unit of the simulation.
     * @returns The length unit to be used in the simulation.
     */
    LengthUnit getLengthUnit() const;

    /**
     * Compute the scale factor to convert micrometers to the geometry's length unit.
     * @returns Scale factor.
     */
    float computeLengthScaleFactor() const;

    /**
     * Set the initial Stokes vector for the rays.
     * @param stokesVector The initial Stokes vector [I, Q, U, V].
     */
    void setStokesVector(const float4 &stokesVector);

    /**
     * Get the initial Stokes vector for the rays.
     * @returns The initial Stokes vector [I, Q, U, V].
     */
    float4 getStokesVector() const { return stokesVector_; }

    /**
     * Set the Q- axis for the Stokes vector.
     * @param qMinusAxis The Q- axis as a float3.
     */
    void setQMinusAxisSeed(const float3 &qMinusAxis) { qMinusAxisSeed_ = qMinusAxis; }

    /**
     * Get the Q- axis for the Stokes vector.
     * @returns The Q- axis as a float3.
     */
    float3 getQMinusAxisSeed() const { return qMinusAxisSeed_; }

    /**
     * Set the maximum scattering count for ray tracing.
     * @param maxScatteringCount The maximum scattering count for ray tracing; if 0, no limit.
     */
    void setMaxScatteringCount(uint32_t maxScatteringCount);

    /**
     * Get the maximum scattering count for ray tracing.
     * @returns Maximum scattering count for ray tracing. If 0, there is no limit.
     */
    uint32_t getMaxScatteringCount() const;

    /**
     * Set the maximum number of nested geometry levels.
     * @param maxNestedGeometryLevels The maximum number of nested geometry levels [1-31].
     */
    void setMaxNestedGeometryLevels(uint32_t maxNestedGeometryLevels);

    /**
     * Get the maximum number of nested geometry levels.
     * @returns Maximum number of nested geometry levels.
     */
    uint32_t getMaxNestedGeometryLevels() const;

    /**
     * Get the maximum number of sub-geometries for a Instance geometry.
     * @returns Maximum number of sub-geometries.
     */
    uint32_t getMaxSubGeometries() const;

    /**
     * Get the maximum number of triangles for a Mesh geometry.
     * @returns Maximum number of triangles.
     */
    uint32_t getMaxMeshTriangles() const;

    /**
     * Get the initial seed for random number generation.
     * @returns Initial seed.
     */
    uint32_t getInitSeed() const;

    /**
     * Set the initial seed for random number generation.
     * @param initSeed The initial seed to be used in the simulation.
     */
    void setInitSeed(uint32_t initSeed);

    /**
     * Set whether to use complex Fresnel equations.
     * @param useComplexFresnel True to use complex Fresnel equations, false otherwise
     */
    void setUseComplexFresnel(bool useComplexFresnel);

    /**
     * Get whether complex Fresnel equations are used.
     * @returns True if complex Fresnel equations are used, false otherwise
     */
    bool getUseComplexFresnel() const;

    void setDirectionHealpixNside(uint32_t nside);

    uint32_t getDirectionHealpixNside() const { return healpixMissNside_; }

    void initializePipeline();

    void initializePipelineDensity();

    void setRayGenerator(std::shared_ptr<IRayGenerator> generator);

    std::shared_ptr<IRayGenerator> getRayGenerator() const;

    void setRayOutput(std::shared_ptr<RayTracingOutput> output)
    {
        rayTracingResult_ = std::move(output);
        buffersDirty_ = true;
    }

    std::shared_ptr<RayTracingOutput> getRayOutput() const
    {
        return rayTracingResult_;
    }

    void allocateOutputBuffers(uint3 launchShape);

    OutputBuffers *getDeviceOutputBuffers() const
    {
        return outputBuffers_.get();
    }

    void freeOutputBuffers();

    void freeDeviceMemory();

    void run();

    float calculateVolumeFraction(float3 boxMin, float3 boxMax, uint32_t numSamples);

private:
    OptixDeviceContext context_ = nullptr;
    std::shared_ptr<IGeometry> geometry_;
    std::shared_ptr<IGeometry> previousGeometry_;

    bool geometryDirty_ = false;
    bool sbtDirty_ = false;
    bool buffersDirty_ = false;
    bool pipelineDirty_ = false;

    LengthUnit lengthUnit_ = MICRO_METER;
    std::vector<Material> materials_;
    std::shared_ptr<IRayGenerator> rayGenerator_;
    float wavelengthUm_ = 0.0f;
    float4 stokesVector_ = make_float4(1.0f, 0.0f, 0.0f, 0.0f); // Stokes vector: [I, Q, U, V]
    float3 qMinusAxisSeed_ = make_float3(NAN, NAN, NAN);
    uint32_t maxScatteringCount_ = UINT32_MAX;
    uint32_t maxNestedGeometryLevels_ = 5;
    uint32_t initSeed_ = 42;
    bool useComplexFresnel_ = true;
    uint32_t healpixMissNside_ = 0;
    uint32_t healpixHistogramBins_ = 0;

    void updateHealpixBufferShape();
    std::unique_ptr<RayTracingPipeline> rayTracingPipeline_;
    std::unique_ptr<VolumeFractionPipeline> densityPipeline_;

    std::shared_ptr<RayTracingOutput> rayTracingResult_;
    std::unique_ptr<OutputBuffers> outputBuffers_;

    int gpuId_ = 0;
    int optixLoggingLevel_ = 1;
};