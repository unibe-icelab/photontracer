// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <memory>
#include <vector>

#include "i_geometry.h"
#include "photontracer.h"

/**
 * @brief Owns the device context, the acceleration structures and the launch pipelines.
 *
 * Simulation holds the settings and the output buffers and delegates everything
 * that touches the ray tracing API to a backend.
 */
class IRaytracingBackend
{
public:
    virtual ~IRaytracingBackend() = default;

    virtual uint32_t getMaxTraversableGraphDepth() const = 0;
    virtual uint32_t getMaxSubGeometries() const = 0;
    virtual uint32_t getMaxMeshTriangles() const = 0;

    virtual void buildGeometry(IGeometry &geometry) = 0;

    virtual bool hasPipeline() const = 0;
    virtual void initializePipeline(const std::vector<Material> &materials, float wavelengthUm, uint32_t maxTraversableGraphDepth) = 0;
    virtual void updateShaderBindingTable(const std::vector<Material> &materials, float wavelengthUm) = 0;
    virtual void resetPipeline() = 0;
    virtual void launch(InputParameters &params, const IGeometry &geometry, uint3 launchShape) = 0;

    virtual bool hasDensityPipeline() const = 0;
    virtual void initializeDensityPipeline(uint32_t maxTraversableGraphDepth) = 0;
    virtual void resetDensityPipeline() = 0;
    virtual void launchDensity(InputParametersSampleDensity &params, const IGeometry &geometry) = 0;
};
