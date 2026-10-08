// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#pragma once

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <cmath>
#include <stdexcept>

#include "photontracer.h"

class IRayGenerator
{
public:
    virtual ~IRayGenerator() = default;
    virtual RayGeneratorType getType() const = 0;
    virtual RayGeneratorData getData() const = 0;
    virtual uint3 getLaunchShape() const = 0;
};

class ParallelRayGenerator : public IRayGenerator
{
public:
    ParallelRayGenerator(uint32_t numberOfRays, float3 origin, float3 direction, float offsetRadius)
        : numberOfRays_(numberOfRays), origin_(origin), direction_(direction), offsetRadius_(offsetRadius)
    {
        direction_ = otk::normalize(direction_);
    }

    RayGeneratorType getType() const override { return RAYGEN_PARALLEL; }

    RayGeneratorData getData() const override
    {
        RayGeneratorData data;
        data.parallel.numberOfRays = numberOfRays_;
        data.parallel.origin = origin_;
        data.parallel.direction = direction_;
        data.parallel.offsetRadius = offsetRadius_;
        return data;
    }

    uint3 getLaunchShape() const override
    {
        return make_uint3(numberOfRays_, 1, 1);
    }

    // Getters/setters for Python access

    uint32_t getNumberOfRays() const { return numberOfRays_; }
    void setNumberOfRays(uint32_t numberOfRays)
    {
        if (numberOfRays == 0)
        {
            throw std::invalid_argument("number_of_rays must be greater than zero");
        }
        numberOfRays_ = numberOfRays;
    }

    float3 getOrigin() const { return origin_; }
    void setOrigin(const float3 &origin) { origin_ = origin; }

    float3 getDirection() const { return direction_; }
    void setDirection(const float3 &direction) { direction_ = direction; }

    float getOffsetRadius() const { return offsetRadius_; }
    void setOffsetRadius(float radius) { offsetRadius_ = radius; }

private:
    uint32_t numberOfRays_ = 1;
    float3 origin_;
    float3 direction_;
    float offsetRadius_;
};

class IsotropicRayGenerator : public IRayGenerator
{
public:
    IsotropicRayGenerator(uint32_t numberOfRays, float3 center, float sourceRadius, float offsetRadius)
        : numberOfRays_(numberOfRays), center_(center), sourceRadius_(sourceRadius), offsetRadius_(offsetRadius) {}

    RayGeneratorType getType() const override { return RAYGEN_ISOTROPIC; }

    RayGeneratorData getData() const override
    {
        RayGeneratorData data;
        data.isotropic.numberOfRays = numberOfRays_;
        data.isotropic.center = center_;
        data.isotropic.sourceRadius = sourceRadius_;
        data.isotropic.offsetRadius = offsetRadius_;
        return data;
    }

    uint3 getLaunchShape() const override
    {
        return make_uint3(numberOfRays_, 1, 1);
    }

    uint32_t getNumberOfRays() const { return numberOfRays_; }

    void setNumberOfRays(uint32_t numberOfRays)
    {
        if (numberOfRays == 0)
        {
            throw std::invalid_argument("number_of_rays must be greater than zero");
        }
        numberOfRays_ = numberOfRays;
    }

    // Getters/setters for Python access
    float3 getCenter() const { return center_; }
    void setCenter(const float3 &center) { center_ = center; }

    float getSourceRadius() const { return sourceRadius_; }
    void setSourceRadius(float radius) { sourceRadius_ = radius; }

    float getOffsetRadius() const { return offsetRadius_; }
    void setOffsetRadius(float radius) { offsetRadius_ = radius; }

private:
    uint32_t numberOfRays_ = 1;
    float3 center_;
    float sourceRadius_;
    float offsetRadius_;
};

class CameraRayGenerator : public IRayGenerator
{
public:
    CameraRayGenerator()
        : aspectRatio_(1.0f),
          imageWidth_(100),
          imageHeight_(100),
          samplesPerPixel_(1),
          verticalFovDeg_(90.0f),
          lookFrom_(make_float3(0.0f, 0.0f, 0.0f)),
          lookAt_(make_float3(0.0f, 0.0f, -1.0f)),
          verticalUp_(make_float3(0.0f, 1.0f, 0.0f)),
          defocusAngleDeg_(0.0f),
          focusDist_(10.0f)
    {
        updateCameraFrame();
    }

    RayGeneratorType getType() const override { return RAYGEN_CAMERA; }

    RayGeneratorData getData() const override
    {
        RayGeneratorData data = {};
        data.camera.center = lookFrom_;
        data.camera.pixel00 = pixel00_;
        data.camera.pixelDeltaU = pixelDeltaU_;
        data.camera.pixelDeltaV = pixelDeltaV_;
        data.camera.defocusDiskU = defocusDiskU_;
        data.camera.defocusDiskV = defocusDiskV_;
        data.camera.imageWidth = imageWidth_;
        data.camera.imageHeight = imageHeight_;
        data.camera.samplesPerPixel = samplesPerPixel_;
        data.camera.enableDefocus = defocusAngleDeg_ > 0.0f ? 1u : 0u;
        return data;
    }

    unsigned int getImageWidth() const { return imageWidth_; }
    void setImageWidth(unsigned int width)
    {
        if (width == 0)
        {
            throw std::invalid_argument("image_width must be greater than zero");
        }
        imageWidth_ = width;
        updateCameraFrame();
    }

    unsigned int getImageHeight() const { return imageHeight_; }
    void setImageHeight(unsigned int height)
    {
        if (height == 0)
        {
            throw std::invalid_argument("image_height must be greater than zero");
        }
        updateCameraFrame(height);
    }

    uint3 getLaunchShape() const override
    {
        return make_uint3(samplesPerPixel_, imageWidth_, imageHeight_);
    }

    float getAspectRatio() const { return aspectRatio_; }
    void setAspectRatio(float ratio)
    {
        if (!(ratio > 0.0f))
        {
            throw std::invalid_argument("aspect_ratio must be greater than zero");
        }
        aspectRatio_ = ratio;
        updateCameraFrame();
    }

    unsigned int getSamplesPerPixel() const { return samplesPerPixel_; }
    void setSamplesPerPixel(unsigned int samplesPerPixel)
    {
        if (samplesPerPixel == 0)
        {
            throw std::invalid_argument("samples_per_pixel must be greater than zero");
        }
        samplesPerPixel_ = samplesPerPixel;
    }

    float getVerticalFov() const { return verticalFovDeg_; }
    void setVerticalFov(float verticalFov)
    {
        if (!(verticalFov > 0.0f))
        {
            throw std::invalid_argument("vertical_fov must be greater than zero");
        }
        verticalFovDeg_ = verticalFov;
        updateCameraFrame();
    }

    float3 getLookFrom() const { return lookFrom_; }
    void setLookFrom(const float3 &lookFrom)
    {
        lookFrom_ = lookFrom;
        updateCameraFrame();
    }

    float3 getLookAt() const { return lookAt_; }
    void setLookAt(const float3 &lookAt)
    {
        lookAt_ = lookAt;
        updateCameraFrame();
    }

    float3 getVerticalUp() const { return verticalUp_; }
    void setVerticalUp(const float3 &verticalUp)
    {
        verticalUp_ = verticalUp;
        updateCameraFrame();
    }

    float getDefocusAngle() const { return defocusAngleDeg_; }
    void setDefocusAngle(float angleDegrees)
    {
        if (angleDegrees < 0.0f)
        {
            throw std::invalid_argument("defocus_angle cannot be negative");
        }
        defocusAngleDeg_ = angleDegrees;
        updateCameraFrame();
    }

    float getFocusDistance() const { return focusDist_; }
    void setFocusDistance(float distance)
    {
        if (!(distance > 0.0f))
        {
            throw std::invalid_argument("focus_distance must be greater than zero");
        }
        focusDist_ = distance;
        updateCameraFrame();
    }

private:
    static float degreesToRadians(float degrees)
    {
        return degrees * M_PI / 180.0f;
    }

    void updateImageDimensions(uint32_t height=0)
    {
        if (height > 0)
        {
            imageHeight_ = height;
            aspectRatio_ = static_cast<float>(imageWidth_) / static_cast<float>(imageHeight_);
        }
        else
        {
            imageHeight_ = static_cast<unsigned int>(std::round(static_cast<float>(imageWidth_) / aspectRatio_));
        }
        if (imageHeight_ == 0)
        {
            imageHeight_ = 1;
        }
    }

    void updateCameraFrame(uint32_t height=0)
    {
        updateImageDimensions(height);

        const float exactAspectRatio = static_cast<float>(imageWidth_) / static_cast<float>(imageHeight_);
        const float theta = degreesToRadians(verticalFovDeg_);
        const float h = tanf(0.5f * theta);
        const float viewportHeight = 2.0f * h * focusDist_;
        const float viewportWidth = viewportHeight * exactAspectRatio;

        float3 w = otk::normalize(lookFrom_ - lookAt_);
        if (!std::isfinite(w.x) || !std::isfinite(w.y) || !std::isfinite(w.z))
        {
            w = make_float3(0.0f, 0.0f, 1.0f);
        }

        float3 up = verticalUp_;
        if (!std::isfinite(up.x) || !std::isfinite(up.y) || !std::isfinite(up.z) || otk::dot(up, up) < 1e-6f)
        {
            up = make_float3(0.0f, 1.0f, 0.0f);
        }

        float3 u = otk::normalize(otk::cross(up, w));
        if (!std::isfinite(u.x) || !std::isfinite(u.y) || !std::isfinite(u.z))
        {
            up = make_float3(1.0f, 0.0f, 0.0f);
            if (fabsf(otk::dot(up, w)) > 0.999f)
            {
                up = make_float3(0.0f, 1.0f, 0.0f);
            }
            u = otk::normalize(otk::cross(up, w));
        }
        float3 v = otk::cross(w, u);

        const float3 viewportU = viewportWidth * u;
        const float3 viewportV = -viewportHeight * v;

        pixelDeltaU_ = viewportU / static_cast<float>(imageWidth_);
        pixelDeltaV_ = viewportV / static_cast<float>(imageHeight_);

        const float3 viewportUpperLeft = lookFrom_ - focusDist_ * w - 0.5f * viewportU - 0.5f * viewportV;
        pixel00_ = viewportUpperLeft + 0.5f * (pixelDeltaU_ + pixelDeltaV_);

        const float defocusRadius = focusDist_ * tanf(0.5f * degreesToRadians(defocusAngleDeg_));
        if (defocusAngleDeg_ > 0.0f && defocusRadius > 0.0f)
        {
            defocusDiskU_ = u * defocusRadius;
            defocusDiskV_ = v * defocusRadius;
        }
        else
        {
            defocusDiskU_ = make_float3(0.0f, 0.0f, 0.0f);
            defocusDiskV_ = make_float3(0.0f, 0.0f, 0.0f);
        }
    }

    float aspectRatio_;
    unsigned int imageWidth_;
    unsigned int imageHeight_;
    unsigned int samplesPerPixel_;
    float verticalFovDeg_;
    float3 lookFrom_;
    float3 lookAt_;
    float3 verticalUp_;
    float defocusAngleDeg_;
    float focusDist_;

    float3 pixelDeltaU_;
    float3 pixelDeltaV_;
    float3 pixel00_;
    float3 defocusDiskU_;
    float3 defocusDiskV_;
};