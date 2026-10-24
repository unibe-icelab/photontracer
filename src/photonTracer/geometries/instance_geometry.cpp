// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <vector>
#include <stdexcept>
#include <string>

#include "../i_geometry.h"
#include "instance_geometry.h"
#include "mesh_geometry.h"

InstanceGeometry::InstanceGeometry(
    std::vector<std::shared_ptr<IGeometry>> subGeometries,
    std::vector<float> instanceTransforms,
    std::vector<unsigned int> particleTypeIds,
    std::vector<unsigned int> materialIds)
    : subGeometries(std::move(subGeometries)),
      instanceTransforms(std::move(instanceTransforms)),
      particleTypeIds(std::move(particleTypeIds)),
      materialIds(std::move(materialIds)) {}

void InstanceGeometry::validateMaterialIds(std::size_t materialCount) const
{
    for (unsigned int materialId : materialIds)
    {
        if (materialId >= materialCount)
        {
            throw std::invalid_argument("Instance material ID " + std::to_string(materialId) + " is out of range for " + std::to_string(materialCount) + " configured materials.");
        }
    }
    for (const auto &subGeometry : subGeometries)
    {
        if (subGeometry && subGeometry->getType() == MESH_INSTANCED)
        {
            subGeometry->validateMaterialIds(materialCount);
        }
    }
}

void InstanceGeometry::freeDeviceMemory()
{
    for (auto &subGeometry : subGeometries)
    {
        if (subGeometry)
        {
            subGeometry->freeDeviceMemory();
        }
    }

    IGeometry::freeDeviceMemory();
}


GeometryType InstanceGeometry::getType() const
{
    return MESH_INSTANCED;
}