// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <vector>
#include <chrono>
#include <stdexcept>
#include <string>

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include "../photontracer.h"
#include "geometry_util.h"
#include "../i_geometry.h"
#include "mesh_geometry.h"

MeshGeometry::MeshGeometry(std::vector<float3> vertices, std::vector<unsigned int> indices, bool compact)
    : vertices(std::move(vertices)), indices(std::move(indices)), compact(compact) {}
void MeshGeometry::build(OptixDeviceContext &context)
{
    buildGasFromMesh(
        vertices, indices, context,
        traversableHandle, dGeometryBuffer, compact);
    accelerationStructureBuilt = true;
}
void MeshGeometry::validateMaterialIds(std::size_t materialCount) const
{
    if (materialCount < 2)
    {
        throw std::invalid_argument("A MeshGeometry needs at least 2 materials (0: outside, 1: inside), got " + std::to_string(materialCount) + ".");
    }
}

GeometryType MeshGeometry::getType() const
{
    return MESH;
}