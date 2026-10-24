// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <cstddef>
#include <memory>

#include "photontracer.h"

/**
 * @brief Acceleration structure of a geometry for one backend.
 *
 * Created by the backend that builds it, which also knows the concrete type.
 * Destroying it releases the memory.
 */
class AccelerationStructure
{
public:
    explicit AccelerationStructure(const void *owner) : owner_(owner) {}
    virtual ~AccelerationStructure() = default;

    AccelerationStructure(const AccelerationStructure &) = delete;
    AccelerationStructure &operator=(const AccelerationStructure &) = delete;

    const void *owner() const { return owner_; }

private:
    const void *owner_; ///< The backend that built this
};

/**
 * @brief Geometry of the scene. It only holds the scene description; the backend
 * builds an acceleration structure from it and attaches it here.
 */
class IGeometry
{
public:
    virtual ~IGeometry() = default;

    /// Releases the acceleration structure, and those of sub-geometries
    virtual void freeDeviceMemory()
    {
        accelerationStructure_.reset();
    }

    bool isBuilt() const
    {
        return accelerationStructure_ != nullptr;
    }

    /// Whether the acceleration structure was built by the given backend
    bool isBuiltBy(const void *backend) const
    {
        return accelerationStructure_ && accelerationStructure_->owner() == backend;
    }

    AccelerationStructure *getAccelerationStructure() const
    {
        return accelerationStructure_.get();
    }

    /// Replaces (and releases) the current acceleration structure
    void setAccelerationStructure(std::unique_ptr<AccelerationStructure> accelerationStructure)
    {
        accelerationStructure_ = std::move(accelerationStructure);
    }

    virtual GeometryType getType() const = 0;

    /**
     * @brief Throws std::invalid_argument if the geometry refers to a material that does not exist.
     *
     * @param materialCount Number of configured materials.
     */
    virtual void validateMaterialIds(std::size_t materialCount) const
    {
        (void)materialCount;
    }

private:
    std::unique_ptr<AccelerationStructure> accelerationStructure_;
};
