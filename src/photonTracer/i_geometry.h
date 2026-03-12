// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once
#include <string>
#include <optix.h>
#include <cuda_runtime.h>
#include "photontracer.h"

class IGeometry
{
public:
    /**
     * @brief Destructor for IGeometry.
     */
    virtual ~IGeometry()
    {
        freeDeviceMemory();
    }

    virtual void freeDeviceMemory()
    {
        if (accelerationStructureBuilt && dGeometryBuffer)
        {
            cudaFree(reinterpret_cast<void *>(dGeometryBuffer));
            dGeometryBuffer = 0;
            traversableHandle = {};
        }
        accelerationStructureBuilt = false;
    }

    /**
     * @brief Builds the acceleration structure for this Geometry on the device.
     *
     * @param context The OptiX device context.
     * @return BuildResult containing the traversable handle and output buffer.
     */
    virtual void build(OptixDeviceContext &context) = 0;

    /**
     * @brief Checks if the acceleration structure has been built.
     *
     * @return True if the acceleration structure is built, false otherwise.
     */
    virtual bool isBuilt() const
    {
        return accelerationStructureBuilt;
    }

    /**
     * @brief Returns the type of geometry.
     *
     * @return A string representing the type of geometry.
     */
    virtual GeometryType getType() const = 0;

    /**
     * @brief Returns the traversable handle for the acceleration structure.
     *
     * @return OptixTraversableHandle The traversable handle.
     */
    OptixTraversableHandle getTraversableHandle() const
    {
        return traversableHandle;
    }

protected:
    GeometryType geometryType;                ///< The type of geometry
    OptixTraversableHandle traversableHandle; ///< The traversable handle for the acceleration structure
    CUdeviceptr dGeometryBuffer;              ///< Device pointer to the output buffer containing the acceleration structure
    bool accelerationStructureBuilt = false;  ///< Flag indicating if the geometry has been built
};