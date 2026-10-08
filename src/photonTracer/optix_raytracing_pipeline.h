// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <optix.h>
#include "photontracer.h"

class OptixRayTracingPipeline
{
public:
    OptixRayTracingPipeline(const OptixDeviceContext context, std::vector<Material> materials, const float wavelengthUm,
                            uint32_t maxTraversableGraphDepth, RandomNumberGenerator generator)
        : generator_(generator)
    {
        initialize(context, materials, wavelengthUm, maxTraversableGraphDepth);
    }
    ~OptixRayTracingPipeline();

    void updateShaderBindingTable(const std::vector<Material> materials, const float wavelengthUm);

    void launch(InputParameters &params, uint3 launchDim);

private:
    OptixPipeline pipeline_ = nullptr;
    OptixPipelineCompileOptions pipelineCompileOptions_ = {};
    bool pipelineCompileOptionsInitialized_ = false;

    OptixModule module_ = nullptr;

    OptixProgramGroup raygenProgGroup_ = nullptr;
    OptixProgramGroup tracedRaygenProgGroup_ = nullptr; ///< the same raygen program with the recording of traced rays
    OptixProgramGroup missProgGroup_ = nullptr;
    OptixProgramGroup hitgroupProgGroup_ = nullptr;

    CUdeviceptr raygenRecord_ = 0;
    CUdeviceptr tracedRaygenRecord_ = 0;
    CUdeviceptr missRecord_ = 0;
    CUdeviceptr hitgroupRecord_ = 0;

    RandomNumberGenerator generator_; ///< selects the kernel module
    uint32_t maxTraversableGraphDepth_ = 1;
    const uint32_t maxTraceDepth_ = 1; // no recursion, we use iterative path tracing

    OptixShaderBindingTable sbt_ = {};
    bool sbtInitialized_ = false;

    void initialize(
        const OptixDeviceContext context, std::vector<Material> materials, 
        const float wavelengthUm, uint32_t maxTraversableGraphDepth);
    void setupPipelineCompileOptions();
    void createModules(const OptixDeviceContext context);
    void createProgramGroups(const OptixDeviceContext context);
    void linkPipeline(const OptixDeviceContext context);
    void setupShaderBindingTable(const std::vector<Material> materials, const float wavelengthUm);
    void cleanupShaderBindingTable();
};

class OptixVolumeFractionPipeline
{
public:
    OptixVolumeFractionPipeline(const OptixDeviceContext context, uint32_t maxTraversableGraphDepth, RandomNumberGenerator generator)
        : generator_(generator)
    {
        initialize(context, maxTraversableGraphDepth);
    }
    ~OptixVolumeFractionPipeline();

    void launch(InputParametersSampleDensity &params);

private:
    OptixPipeline pipeline_ = nullptr;
    OptixPipelineCompileOptions pipelineCompileOptions_ = {};
    bool pipelineCompileOptionsInitialized_ = false;

    OptixModule module_ = nullptr;

    OptixProgramGroup raygenProgGroup_ = nullptr;
    OptixProgramGroup missProgGroup_ = nullptr;
    OptixProgramGroup hitgroupProgGroup_ = nullptr;

    CUdeviceptr raygenRecord_ = 0;
    CUdeviceptr missRecord_ = 0;
    CUdeviceptr hitgroupRecord_ = 0;

    RandomNumberGenerator generator_; ///< selects the kernel module
    uint32_t maxTraversableGraphDepth_ = 1;
    const uint32_t maxTraceDepth_ = 1; // no recursion, we use iterative path tracing

    OptixShaderBindingTable sbt_ = {};
    bool sbtInitialized_ = false;

    void initialize(
        const OptixDeviceContext context, uint32_t maxTraversableGraphDepth);
    void setupPipelineCompileOptions();
    void createModules(const OptixDeviceContext context);
    void createProgramGroups(const OptixDeviceContext context);
    void linkPipeline(const OptixDeviceContext context);
    void setupShaderBindingTable();
    void cleanupShaderBindingTable();
};