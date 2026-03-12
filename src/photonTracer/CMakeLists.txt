include(GNUInstallDirs)
include(BuildConfig)
include(embed_cuda)

add_library(Geometry
  geometries/geometry_util.cpp
  geometries/instance_geometry.cpp
  geometries/mesh_geometry.cpp
)
target_compile_options(Geometry PRIVATE -fPIC)

target_link_libraries(Geometry PUBLIC 
  OptiX::OptiX
  OptiXToolkit::Memory
  OptiXToolkit::ShaderUtil
  CUDA::cuda_driver)

embed_cuda(
  CONST HEADER photontracer_kernels.h
  OUTPUT_TARGET
    photonTracerKernels
  LIBRARIES
    OptiXToolkit::ShaderUtil
    OptiXToolkit::Memory
    OptiX::OptiX
  SOURCES
    photontracer.cu
  INCLUDES
    ${CMAKE_CURRENT_LIST_DIR}
)

pybind11_add_module( photontracer_bindings 
  photontracer_bindings.cpp
  simulation.cpp
  raytracing_pipeline.cpp
  raytracing_output.cpp 
  output_buffers.cpp
  )
  
install(TARGETS photontracer_bindings
        LIBRARY DESTINATION photontracer)

otk_target_options( photontracer_bindings )

target_include_directories(photontracer_bindings PRIVATE
  ${CMAKE_BINARY_DIR}/src/photonTracer/photonTracerKernels
)

target_link_libraries( photontracer_bindings PRIVATE
  Geometry
  photonTracerKernels
  OptiX::OptiX
  OptiXToolkit::Memory
  OptiXToolkit::ShaderUtil
  CUDA::cuda_driver
  CUDA::cudart_static
  )
