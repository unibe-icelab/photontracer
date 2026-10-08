// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <embree4/rtcore.h>

#include "geometries/instance_geometry.h"
#include "geometries/mesh_geometry.h"
#include "i_raytracing_backend.h"
#include "spawn_point.h"

class EmbreeAccelerationStructure;

/// An instance in the scene of an InstanceGeometry; Embree numbers them in the order they were added
struct EmbreeInstance
{
    InstanceTransform transform;
    unsigned int materialId;
    const EmbreeAccelerationStructure *child;
};

/// Embree scene of a geometry. A mesh reads the triangles of a hit from the mesh it was built
/// from, an InstanceGeometry looks up the instance and the scene a hit belongs to.
class EmbreeAccelerationStructure : public AccelerationStructure
{
public:
    using AccelerationStructure::AccelerationStructure;
    ~EmbreeAccelerationStructure() override
    {
        if (scene)
        {
            rtcReleaseScene(scene);
        }
    }

    RTCScene scene = nullptr;
    const MeshGeometry *mesh = nullptr;
    std::vector<EmbreeInstance> instances;
    unsigned int instanceLevels = 0; ///< how deep instances are nested in this scene
};

/**
 * @brief Traces on the CPU with Embree. Every ray is one task of a TBB loop; the
 * output buffers are plain host memory.
 */
class EmbreeRaytracingBackend : public IRaytracingBackend
{
public:
    /// @param cpuThreads Number of threads; 0 uses all cores.
    explicit EmbreeRaytracingBackend(uint32_t cpuThreads = 0);
    ~EmbreeRaytracingBackend() override;

    EmbreeRaytracingBackend(const EmbreeRaytracingBackend &) = delete;
    EmbreeRaytracingBackend &operator=(const EmbreeRaytracingBackend &) = delete;

    uint32_t getMaxTraversableGraphDepth() const override;
    uint32_t getMaxSubGeometries() const override;
    uint32_t getMaxMeshTriangles() const override;

    void *allocateBuffer(size_t bytes) override;
    void freeBuffer(void *buffer) noexcept override;
    void clearBuffer(void *buffer, size_t bytes) override;
    void copyBufferToHost(void *host, const void *buffer, size_t bytes) override;

    void buildGeometry(IGeometry &geometry) override;

    bool supportsRandomNumberGenerator(RandomNumberGenerator generator) const override
    {
        return generator == RandomNumberGenerator::PCG32;
    }

    bool hasPipeline() const override { return hasPipeline_; }
    void initializePipeline(const std::vector<Material> &materials, float wavelengthUm, uint32_t maxTraversableGraphDepth,
                            RandomNumberGenerator generator) override;
    void updateShaderBindingTable(const std::vector<Material> &materials, float wavelengthUm) override;
    void resetPipeline() override { hasPipeline_ = false; }
    void launch(InputParameters &params, const IGeometry &geometry, uint3 launchShape) override;

    bool hasDensityPipeline() const override { return hasDensityPipeline_; }
    void initializeDensityPipeline(uint32_t maxTraversableGraphDepth, RandomNumberGenerator generator) override;
    void resetDensityPipeline() override { hasDensityPipeline_ = false; }
    void launchDensity(InputParametersSampleDensity &params, const IGeometry &geometry) override;

private:
    const EmbreeAccelerationStructure &accelerationStructure(const IGeometry &geometry) const;
    void buildMesh(const MeshGeometry &mesh, EmbreeAccelerationStructure &structure);
    void buildInstances(const InstanceGeometry &instances, EmbreeAccelerationStructure &structure);
    void checkDeviceError(const char *what) const;

    RTCDevice device_ = nullptr;
    uint32_t cpuThreads_;
    HitGroupData hitGroup_ = {};
    bool hasPipeline_ = false;
    bool hasDensityPipeline_ = false;
};
