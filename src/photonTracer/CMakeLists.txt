include(GNUInstallDirs)

# Settings, output buffers, geometry data and ray generators; plain C++ without any backend
add_library(PhotonTracerCore STATIC
  simulation.cpp
  output_buffers.cpp
  raytracing_output.cpp
  geometries/instance_geometry.cpp
  geometries/mesh_geometry.cpp
)
target_include_directories(PhotonTracerCore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT PHOTONTRACER_BUILD_OPTIX)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(PhotonTracerCore PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  return()
endif()

include(BuildConfig)
include(embed_cuda)

target_link_libraries(PhotonTracerCore PUBLIC
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

add_library(PhotonTracerOptix STATIC
  optix_raytracing_backend.cpp
  optix_raytracing_pipeline.cpp
  optix_geometry_build.cpp
)

target_include_directories(PhotonTracerOptix PRIVATE
  ${CMAKE_BINARY_DIR}/src/photonTracer/photonTracerKernels
)

target_link_libraries(PhotonTracerOptix PUBLIC
  PhotonTracerCore
  photonTracerKernels
  OptiX::OptiX
  OptiXToolkit::Memory
  OptiXToolkit::ShaderUtil
  CUDA::cuda_driver
  CUDA::cudart_static
)

pybind11_add_module(photontracer_bindings
  photontracer_bindings.cpp
  backend_factory.cpp
)

install(TARGETS photontracer_bindings
        LIBRARY DESTINATION photontracer)

otk_target_options(photontracer_bindings)

# GCC may emit a false positive in pybind11 internals (-Wstringop-overread).
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(photontracer_bindings PRIVATE -Wno-stringop-overread)
  target_link_options(photontracer_bindings PRIVATE -Wno-stringop-overread)
endif()

target_link_libraries(photontracer_bindings PRIVATE
  PhotonTracerCore
  PhotonTracerOptix
)
