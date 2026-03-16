// This source code is licensed under the BSD-3 license found in the LICENSE file in the root directory of this source tree.
// © 2024-2026, University of Bern, Space Research and Planetary Sciences, Physics Institute, Rafael Ottersberg

#include <optix.h>
#include "photontracer.h"

class RayTracingPipeline
{
public:
    RayTracingPipeline(const OptixDeviceContext context, std::vector<Material> materials, const float wavelengthUm, uint32_t maxTraversableGraphDepth = 1)
    {
        initialize(context, materials, wavelengthUm, maxTraversableGraphDepth);
    }
    ~RayTracingPipeline();

    void updateShaderBindingTable(const std::vector<Material> materials, const float wavelengthUm);

    void launch(InputParameters &params, uint3 launchDim);

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

class VolumeFractionPipeline
{
public:
    VolumeFractionPipeline(const OptixDeviceContext context, uint32_t maxTraversableGraphDepth = 1)
    {
        initialize(context, maxTraversableGraphDepth);
    }
    ~VolumeFractionPipeline();

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