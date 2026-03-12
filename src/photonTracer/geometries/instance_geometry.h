// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <memory>
#include <string>

#include <optix.h>
#include <cuda_runtime.h>

#include "../i_geometry.h"
#include "mesh_geometry.h"

/**
 * @brief Geometry implementation for instance geometry, which allows for instancing of multiple sub-geometries with different transformations.
 * Each instance can have its own transformation matrix, particle type ID, and material ID, but they all share the same underlying geometry data. This is useful for efficiently simulating large numbers of similar objects
 */
class InstanceGeometry : public IGeometry
{
public:
    /**
     * @brief Construct a new InstanceGeometry object.
     *
     * @param numberOfParticles Number of subGeometries in the instanced mesh.
     * @param subGeometries Vector of geometry objects.
     * @param instanceTransforms Vector of instance transformation matrices.
     * Each matrix is a 3x4 transformation matrix stored as 12 floats.
     * @param particleTypeIds Shared identifiers for the particle type.
     * @param materialIds Shared identifiers for the material.
     */
    InstanceGeometry(std::vector<std::shared_ptr<IGeometry>> subGeometries,
                     std::vector<float> instanceTransforms,
                     std::vector<unsigned int> particleTypeIds,
                     std::vector<unsigned int> materialIds);

    /**
     * @brief Build the OptiX acceleration structure.
     *
     * @param context The OptiX device context.
     * @return BuildResult containing the traversable handle and output buffer.
     */
    void build(OptixDeviceContext &context) override;

    void freeDeviceMemory() override;

    /**
     * @brief Get the type name of this geometry.
     *
     * @return std::string Type string ("InstanceGeometry").
     */
    GeometryType getType() const override;

private:
    std::vector<std::shared_ptr<IGeometry>> subGeometries;
    std::vector<float> instanceTransforms;
    std::vector<unsigned int> particleTypeIds;
    std::vector<unsigned int> materialIds;
};