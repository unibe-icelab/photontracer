// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <vector>
#include <string>

#include <optix.h>
#include <cuda_runtime.h>

#include "../photontracer.h"
#include "../i_geometry.h"

/**
 * @brief Geometry implementation for a triangulate mesh.
 */
class MeshGeometry : public IGeometry
{
public:
    /**
     * @brief Construct a new MeshGeometry object
     *
     * @param vertices List of vertex positions.
     * @param indices List of triangle indices.
     */
    MeshGeometry(std::vector<float3> vertices, std::vector<unsigned int> indices);

    /**
     * @brief Build the OptiX acceleration structure for the mesh.
     *
     * @param context The OptiX device context.
     * @return BuildResult containing the traversable handle and output buffer.
     */
    void build(OptixDeviceContext &context) override;

    /**
     * @brief Get the type name of this geometry.
     *
     * @return std::string Type string ("MeshGeometry").
     */
    GeometryType getType() const override;

private:
    std::vector<float3> vertices;      ///< Vertex positions of the mesh
    std::vector<unsigned int> indices; ///< Indices defining the mesh triangle faces
};