// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <memory>

#include <optix.h>

#include "i_raytracing_backend.h"
#include "optix_raytracing_pipeline.h"

class OptixRaytracingBackend : public IRaytracingBackend
{
public:
    OptixRaytracingBackend(int gpuId, int optixLoggingLevel, bool enableValidationMode);
    ~OptixRaytracingBackend() override;

    OptixRaytracingBackend(const OptixRaytracingBackend &) = delete;
    OptixRaytracingBackend &operator=(const OptixRaytracingBackend &) = delete;

    uint32_t getMaxTraversableGraphDepth() const override;
    uint32_t getMaxSubGeometries() const override;
    uint32_t getMaxMeshTriangles() const override;

    void buildGeometry(IGeometry &geometry) override;

    bool hasPipeline() const override { return pipeline_ != nullptr; }
    void initializePipeline(const std::vector<Material> &materials, float wavelengthUm, uint32_t maxTraversableGraphDepth) override;
    void updateShaderBindingTable(const std::vector<Material> &materials, float wavelengthUm) override;
    void resetPipeline() override { pipeline_.reset(); }
    void launch(InputParameters &params, const IGeometry &geometry, uint3 launchShape) override;

    bool hasDensityPipeline() const override { return densityPipeline_ != nullptr; }
    void initializeDensityPipeline(uint32_t maxTraversableGraphDepth) override;
    void resetDensityPipeline() override { densityPipeline_.reset(); }
    void launchDensity(InputParametersSampleDensity &params, const IGeometry &geometry) override;

private:
    uint32_t getDeviceLimit(OptixDeviceProperty property) const;
    static void contextLogCb(uint32_t level, const char *tag, const char *message, void *cbdata);

    OptixDeviceContext context_ = nullptr;
    std::unique_ptr<OptixRayTracingPipeline> pipeline_;
    std::unique_ptr<OptixVolumeFractionPipeline> densityPipeline_;
};
