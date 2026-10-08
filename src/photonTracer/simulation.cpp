// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

// Portions of this code were derived from NVIDIA OptiX sample code.
// See THIRD_PARTY_NOTICES.md for full license text.

#include <iomanip>
#include <cmath>
#include <limits>

#include <optix.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include "simulation.h"
#include "raytracing_output.h"
#include "output_buffers.h"

Simulation::Simulation(int gpuId, int optixLoggingLevel, bool enableValidationMode)
{
    initializeContext(gpuId, optixLoggingLevel, enableValidationMode);
}

Simulation::~Simulation()
{
    freeDeviceMemory();

    if (context_)
    {
        optixDeviceContextDestroy(context_);
    }
    cudaDeviceReset();
}

void Simulation::initializeContext(int gpuId, int optixLoggingLevel, bool enableValidationMode)
{
    gpuId_ = gpuId;
    optixLoggingLevel_ = optixLoggingLevel;
    // Set the CUDA device before initializing
    int deviceCount = 0;
    OTK_ERROR_CHECK(cudaGetDeviceCount(&deviceCount));

    if (gpuId_ >= 0 && gpuId_ < deviceCount)
    {
        std::cout << "Selecting GPU device ID: " << gpuId_ + 1
                  << "/" << deviceCount << std::endl;
        OTK_ERROR_CHECK(cudaSetDevice(gpuId_));

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gpuId_);
        std::cout << "Using device: " << prop.name << std::endl;
    }
    else
    {
        std::cerr << "Warning: Invalid device ID " << gpuId_
                  << ". Using default device." << std::endl;
    }

    // Initialize CUDA context
    OTK_ERROR_CHECK(cudaFree(0));

    // Initialize the OptiX API, loading all API entry points
    OTK_ERROR_CHECK(optixInit());

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &contextLogCb;
    options.logCallbackLevel = optixLoggingLevel_;

    if (enableValidationMode)
    {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    }

    // Associate a CUDA context (current) with this OptiX context
    CUcontext cuCtx = 0; // zero means take the current context
    OTK_ERROR_CHECK(optixDeviceContextCreate(cuCtx, &options, &context_));
}

// Callback function signature for OptiX logging
void Simulation::contextLogCb(uint32_t level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

void Simulation::setLengthUnit(LengthUnit lengthUnit)
{
    lengthUnit_ = lengthUnit;
}

LengthUnit Simulation::getLengthUnit() const
{
    return lengthUnit_;
}

void Simulation::setGeometry(std::shared_ptr<IGeometry> geometry)
{
    if (!geometry)
    {
        throw std::invalid_argument("Geometry pointer must not be null");
    }

    if (geometry_ != geometry)
    {
        // If geometry type has changed, pipeline needs to be rebuilt
        if (geometry_ && geometry->getType() != geometry_->getType())
        {
            pipelineDirty_ = true;
        }
        previousGeometry_ = geometry_;
        geometry_ = std::move(geometry);
        geometryDirty_ = true;
    }
}

std::shared_ptr<IGeometry> Simulation::getGeometry() const
{
    if (!geometry_)
    {
        throw std::runtime_error("Geometry not set");
    }
    return geometry_;
}

float Simulation::computeLengthScaleFactor() const
{
    // return factor to convert micrometers to stored geometry units
    if (lengthUnit_ == MICRO_METER)
    {
        return 1.0f; // micrometer -> micrometer
    }
    else if (lengthUnit_ == MILLI_METER)
    {
        return 1e-3f; // micrometer -> millimeter
    }
    else if (lengthUnit_ == METER)
    {
        return 1e-6f; // micrometer -> meter
    }
    else
    {
        throw std::runtime_error("Invalid length unit");
    }
}

void Simulation::setWavelengthUm(float wavelengthUm)
{
    wavelengthUm_ = wavelengthUm;
    sbtDirty_ = true;
}

float Simulation::getWavelengthUm() const
{
    return wavelengthUm_;
}

void Simulation::setMaterials(const std::vector<Material> &materials)
{
    materials_ = materials;
    sbtDirty_ = true;
}

std::vector<Material> Simulation::getMaterials() const
{
    return materials_;
}

void Simulation::setRayGenerator(std::shared_ptr<IRayGenerator> generator)
{
    if (!generator)
    {
        throw std::invalid_argument("Ray generator must not be null");
    }
    rayGenerator_ = std::move(generator);
}

std::shared_ptr<IRayGenerator> Simulation::getRayGenerator() const
{
    return rayGenerator_;
}

void Simulation::setStokesVector(const float4 &stokesVector)
{
    if (stokesVector.x != 1.0f)
    {
        throw std::runtime_error("Error: Stokes vector I must be 1.0f");
    }
    if ((stokesVector.y < -1.0f || stokesVector.y > 1.0f) ||
        (stokesVector.z < -1.0f || stokesVector.z > 1.0f) ||
        (stokesVector.w < -1.0f || stokesVector.w > 1.0f))
    {
        throw std::runtime_error("Error: Stokes vector Q, U, V must be in the range [-1.0f, 1.0f]");
    }
    if (stokesVector.y * stokesVector.y +
            stokesVector.z * stokesVector.z +
            stokesVector.w * stokesVector.w >
        1.0f)
    {
        throw std::runtime_error("Error: The degree of polarization (sqrt(Q^2 + U^2 + V^2)) must be <= I");
    }
    if ((std::isnan(stokesVector.y) || std::isnan(stokesVector.z)) && stokesVector.w != 0.0f)
    {
        throw std::runtime_error("Error: Stokes vector V cannot be non-zero for random linear polarization");
    }
    if (std::isnan(stokesVector.w) && (stokesVector.y != 0.0f || stokesVector.z != 0.0f))
    {
        throw std::runtime_error("Error: Stokes vector Q and U cannot be non-zero for random circular polarization");
    }
    stokesVector_ = stokesVector;
}

void Simulation::setMaxScatteringCount(uint32_t maxScatteringCount)
{
    maxScatteringCount_ = maxScatteringCount;
}

uint32_t Simulation::getMaxScatteringCount() const
{
    return maxScatteringCount_;
}

void Simulation::setMaxNestedGeometryLevels(uint32_t maxNestedGeometryLevels)
{
    uint32_t maxTraversableGraphDepth;
    optixDeviceContextGetProperty(
        context_,
        OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH,
        &maxTraversableGraphDepth,
        sizeof(maxTraversableGraphDepth));

    if (maxNestedGeometryLevels < 1 || maxNestedGeometryLevels > maxTraversableGraphDepth)
    {
        throw std::runtime_error("Error: maxNestedGeometryLevels must be in the range [1, " + std::to_string(maxTraversableGraphDepth) + "]");
    }
    maxNestedGeometryLevels_ = maxNestedGeometryLevels;
}
uint32_t Simulation::getMaxNestedGeometryLevels() const
{
    return maxNestedGeometryLevels_;
}

uint32_t Simulation::getMaxSubGeometries() const
{
    uint32_t value = 0;
    optixDeviceContextGetProperty(
        context_,
        OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,
        &value,
        sizeof(value));
    return value;
}

uint32_t Simulation::getMaxMeshTriangles() const
{
    uint32_t value = 0;
    optixDeviceContextGetProperty(
        context_,
        OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
        &value,
        sizeof(value));
    return value;
}

void Simulation::setInitSeed(uint32_t initSeed)
{
    initSeed_ = initSeed;
}

uint32_t Simulation::getInitSeed() const
{
    return initSeed_;
}

void Simulation::setUseComplexFresnel(bool useComplexFresnel)
{
    useComplexFresnel_ = useComplexFresnel;
}

bool Simulation::getUseComplexFresnel() const
{
    return useComplexFresnel_;
}

void Simulation::setDirectionHealpixNside(uint32_t nside)
{
    // nside must be a power of 2
    if (nside > 0 && (nside & (nside - 1)) != 0)
    {
        throw std::runtime_error("Error: healpix_miss_nside must be a power of 2");
    }
    healpixMissNside_ = nside;
    buffersDirty_ = true;
}

void Simulation::updateHealpixBufferShape()
{
    const uint64_t numberOfHealpixBins = (healpixMissNside_ > 0) ? 12ull * static_cast<uint64_t>(healpixMissNside_) * static_cast<uint64_t>(healpixMissNside_) : 0ull;

    if (numberOfHealpixBins > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
    {
        throw std::runtime_error("healpix_miss_nside is too large for 32-bit Healpix bins");
    }

    const size_t elementCount = static_cast<size_t>(numberOfHealpixBins);

    const bool shapeChanged = (elementCount != healpixHistogramBins_);

    if (!shapeChanged)
    {
        return;
    }

    healpixHistogramBins_ = static_cast<uint32_t>(numberOfHealpixBins);

    if (rayTracingResult_)
    {
        rayTracingResult_->setBufferDimX(OutputType::DIRECTION_HISTOGRAM_HEALPIX, elementCount);
    }

    buffersDirty_ = true;
}

void Simulation::initializePipeline()
{
    if (!context_)
    {
        throw std::runtime_error("OptiX context not initialized. Call initializeContext() first.");
    }
    if (!geometry_)
    {
        throw std::runtime_error("Geometry not set. Call setGeometry() first.");
    }
    if (materials_.empty())
    {
        throw std::runtime_error("Materials not set. Call setMaterials() first.");
    }
    if (wavelengthUm_ <= 0.0f)
    {
        throw std::runtime_error("Wavelength not set. Call setWavelengthUm() first.");
    }
    rayTracingPipeline_ = std::make_unique<RayTracingPipeline>(context_, materials_, wavelengthUm_, maxNestedGeometryLevels_);
}

void Simulation::initializePipelineDensity()
{
    if (!context_)
    {
        throw std::runtime_error("OptiX context not initialized. Call initializeContext() first.");
    }
    if (!geometry_)
    {
        throw std::runtime_error("Geometry not set. Call setGeometry() first.");
    }
    densityPipeline_ = std::make_unique<VolumeFractionPipeline>(context_, maxNestedGeometryLevels_);
}

void Simulation::allocateOutputBuffers(const uint3 launchShape)
{
    const uint64_t totalRays = static_cast<uint64_t>(launchShape.x) * launchShape.y * launchShape.z;
    if (totalRays == 0)
    {
        throw std::runtime_error("Number of rays is zero. Call setNumberOfRays() first.");
    }

    if (!rayTracingResult_)
    {
        throw std::runtime_error("Output not configured. Call setRayOutput() first.");
    }

    if (rayTracingResult_->isOutputEnabled(OutputType::DIRECTION_HISTOGRAM_HEALPIX))
    {
        if (healpixMissNside_ == 0)
        {
            throw std::runtime_error("healpix_miss_nside must be > 0 when MISS_HEALPIX_HISTOGRAM output is enabled");
        }
        updateHealpixBufferShape();
    }
    freeOutputBuffers();
    outputBuffers_ = std::make_unique<OutputBuffers>(launchShape, rayTracingResult_->getDescriptors());
}

void Simulation::freeOutputBuffers()
{
    outputBuffers_.reset();
}

void Simulation::freeDeviceMemory()
{
    freeOutputBuffers();

    if (geometry_)
    {
        geometry_->freeDeviceMemory();
    }
    if (previousGeometry_)
    {
        previousGeometry_->freeDeviceMemory();
    }

    geometry_.reset();
    previousGeometry_.reset();
    rayTracingPipeline_.reset();
}

void Simulation::run()
{
    if (!context_)
    {
        throw std::runtime_error("OptiX context not initialized. Call initializeContext() first.");
    }
    if (!geometry_)
    {
        throw std::runtime_error("Geometry not set. Call setGeometry() first.");
    }
    if (materials_.empty())
    {
        throw std::runtime_error("Materials not set. Call setMaterials() first.");
    }
    if (!rayGenerator_)
    {
        throw std::runtime_error("Ray generator not set. Call setRayGenerator() first.");
    }
    if (!rayTracingPipeline_ || pipelineDirty_)
    {
        initializePipeline();
        pipelineDirty_ = false;
        sbtDirty_ = false; // The new pipeline is already up-to-date
    }
    else if (sbtDirty_)
    {
        rayTracingPipeline_->updateShaderBindingTable(materials_, wavelengthUm_);
        sbtDirty_ = false;
    }

    if (previousGeometry_)
    {
        previousGeometry_->freeDeviceMemory();
        previousGeometry_.reset();
    }

    if (geometryDirty_)
    {
        geometry_->build(context_);
        geometryDirty_ = false;
    }

    RayGeneratorType rayGenType = rayGenerator_->getType();
    RayGeneratorData generatorData = rayGenerator_->getData();
    uint3 launchShape = rayGenerator_->getLaunchShape();

    bool reallocationNeeded = !outputBuffers_;
    if (outputBuffers_)
    {
        const uint3 &configuredLaunchShape = outputBuffers_->getConfiguredLaunchShape();
        reallocationNeeded = configuredLaunchShape != launchShape;
    }

    if (buffersDirty_ || reallocationNeeded)
    {
        allocateOutputBuffers(launchShape);
        buffersDirty_ = false;
    }

    if (outputBuffers_)
    {
        outputBuffers_->clearZeroInitializedBuffers();
    }

    InputParameters params;
    params.stokesVector = stokesVector_;
    params.qMinusAxisSeed = qMinusAxisSeed_;
    params.maxScatteringCount = maxScatteringCount_;
    params.initSeed = initSeed_;
    params.geometryType = geometry_->getType();
    params.lengthScale = computeLengthScaleFactor();
    params.handle = geometry_->getTraversableHandle();
    params.rayGeneratorType = rayGenType;
    params.rayGeneratorData = generatorData;
    params.deviceOutputBuffers = outputBuffers_->getDeviceOutputBuffers();
    params.outputFlags = rayTracingResult_->getOutputFlags();
    params.healpixNside = healpixMissNside_;
    params.healpixBinCount = healpixHistogramBins_;
    params.useComplexFresnel = useComplexFresnel_;

    // Launch the ray tracing pipeline
    rayTracingPipeline_->launch(params, launchShape);
}

float Simulation::calculateVolumeFraction(float3 boxMin, float3 boxMax, uint32_t numSamples)
{
    if (!context_)
    {
        throw std::runtime_error("OptiX context not initialized. Call initializeContext() first.");
    }
    if (!geometry_)
    {
        throw std::runtime_error("Geometry not set. Call setGeometry() first.");
    }
    if (!densityPipeline_)
    {
        initializePipelineDensity();
        pipelineDirty_ = false;
    }
    if (previousGeometry_)
    {
        previousGeometry_->freeDeviceMemory();
        previousGeometry_.reset();
    }
    if (geometryDirty_)
    {
        geometry_->build(context_);
        geometryDirty_ = false;
    }

    // allocate device buffer for counting intersections
    int32_t *dIntersectionCountBuffer = nullptr;
    OTK_ERROR_CHECK(cudaMalloc(reinterpret_cast<void **>(&dIntersectionCountBuffer), sizeof(int32_t) * numSamples)); // Assuming max 1024 bins

    InputParametersSampleDensity params;
    params.initSeed = initSeed_;
    params.numberOfRays = numSamples;
    params.boxMin = boxMin;
    params.boxMax = boxMax;
    params.handle = geometry_->getTraversableHandle();
    params.intersectionCountBuffer = dIntersectionCountBuffer;

    // Launch the ray tracing pipeline
    densityPipeline_->launch(params);

    // Copy the intersection count back to host
    std::vector<int32_t> hIntersectionCounts(numSamples);
    OTK_ERROR_CHECK(cudaMemcpy(hIntersectionCounts.data(), dIntersectionCountBuffer, sizeof(int32_t) * numSamples, cudaMemcpyDeviceToHost));

    uint32_t countInside = 0;
    for (int32_t &count : hIntersectionCounts)
    {
        if (count > 0)
        {
            countInside++;
        }
    }

    float volumeFraction = static_cast<float>(countInside) / static_cast<float>(numSamples);

    // Free device buffer
    OTK_ERROR_CHECK(cudaFree(dIntersectionCountBuffer));
    densityPipeline_.reset();

    return volumeFraction;
}
