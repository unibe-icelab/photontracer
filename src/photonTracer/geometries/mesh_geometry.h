// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <vector>
#include <string>

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
     * @param compact Compact the acceleration structure to save device memory.
     */
    MeshGeometry(std::vector<float3> vertices, std::vector<unsigned int> indices, bool compact = true);

    /**
     * @brief Get the type name of this geometry.
     *
     * @return std::string Type string ("MeshGeometry").
     */
    GeometryType getType() const override;

    void validateMaterialIds(std::size_t materialCount) const override;

    const std::vector<float3> &getVertices() const { return vertices; }
    const std::vector<unsigned int> &getIndices() const { return indices; }
    bool isCompact() const { return compact; }

private:
    std::vector<float3> vertices;      ///< Vertex positions of the mesh
    std::vector<unsigned int> indices; ///< Indices defining the mesh triangle faces
    bool compact;                      ///< Whether the acceleration structure is compacted after the build
};