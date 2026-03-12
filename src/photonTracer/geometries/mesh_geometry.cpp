// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <vector>
#include <chrono>

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include "../photontracer.h"
#include "geometry_util.h"
#include "../i_geometry.h"
#include "mesh_geometry.h"

MeshGeometry::MeshGeometry(std::vector<float3> vertices, std::vector<unsigned int> indices)
    : vertices(std::move(vertices)), indices(std::move(indices)) {}
void MeshGeometry::build(OptixDeviceContext &context)
{
    buildGasFromMesh(
        vertices, indices, context,
        traversableHandle, dGeometryBuffer);
    accelerationStructureBuilt = true;
}
GeometryType MeshGeometry::getType() const
{
    return MESH;
}