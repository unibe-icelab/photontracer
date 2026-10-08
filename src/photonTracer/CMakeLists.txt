include(GNUInstallDirs)

# Settings, output buffers, geometry data and ray generators; plain C++ without any backend
add_library(PhotonTracerCore STATIC
  simulation.cpp
  trace_buffers.cpp
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
endif()

if(PHOTONTRACER_BUILD_EMBREE)
  add_library(PhotonTracerEmbree STATIC embree_raytracing_backend.cpp)
  target_link_libraries(PhotonTracerEmbree PUBLIC PhotonTracerCore embree TBB::tbb)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(PhotonTracerEmbree PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

if(PHOTONTRACER_BUILD_OPTIX)
include(BuildConfig)
include(embed_cuda)

target_link_libraries(PhotonTracerCore PUBLIC
  OptiX::OptiX
  OptiXToolkit::Memory
  OptiXToolkit::ShaderUtil
  CUDA::cuda_driver)

# The kernels are compiled once per random number generator, and the backend picks one at run time
foreach(RNG IN ITEMS PCG32 MRG32K3A)
  if(RNG STREQUAL "PCG32")
    set(NAME Pcg32)
    set(FILE pcg32)
  else()
    set(NAME Mrg32k3a)
    set(FILE mrg32k3a)
  endif()

  embed_cuda(
    CONST HEADER photontracer_kernels_${FILE}.h
    OUTPUT_TARGET
      photonTracerKernels${NAME}
    LIBRARIES
      OptiXToolkit::ShaderUtil
      OptiXToolkit::Memory
      OptiX::OptiX
    SOURCES
      photontracer.cu
    EMBEDDED_SYMBOL_NAMES
      photontracer${NAME}Cuda
    INCLUDES
      ${CMAKE_CURRENT_LIST_DIR}
  )
  target_compile_definitions(photonTracerKernels${NAME}Cuda PRIVATE PHOTONTRACER_RNG_${RNG})
endforeach()

add_library(PhotonTracerOptix STATIC
  optix_raytracing_backend.cpp
  optix_raytracing_pipeline.cpp
  optix_geometry_build.cpp
)

target_include_directories(PhotonTracerOptix PRIVATE
  ${CMAKE_BINARY_DIR}/src/photonTracer/photonTracerKernelsPcg32
  ${CMAKE_BINARY_DIR}/src/photonTracer/photonTracerKernelsMrg32k3a
)

target_link_libraries(PhotonTracerOptix PUBLIC
  PhotonTracerCore
  photonTracerKernelsPcg32
  photonTracerKernelsMrg32k3a
  OptiX::OptiX
  OptiXToolkit::Memory
  OptiXToolkit::ShaderUtil
  CUDA::cuda_driver
  CUDA::cudart_static
)

endif()

if(NOT PHOTONTRACER_BUILD_OPTIX AND NOT PHOTONTRACER_BUILD_EMBREE)
  return()
endif()

pybind11_add_module(photontracer_bindings
  photontracer_bindings.cpp
  backend_factory.cpp
)

install(TARGETS photontracer_bindings
        LIBRARY DESTINATION photontracer)

if(PHOTONTRACER_BUILD_OPTIX)
  otk_target_options(photontracer_bindings)
endif()

# GCC may emit a false positive in pybind11 internals (-Wstringop-overread).
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(photontracer_bindings PRIVATE -Wno-stringop-overread)
  target_link_options(photontracer_bindings PRIVATE -Wno-stringop-overread)
endif()

target_link_libraries(photontracer_bindings PRIVATE PhotonTracerCore)
if(PHOTONTRACER_BUILD_OPTIX)
  target_link_libraries(photontracer_bindings PRIVATE PhotonTracerOptix)
endif()
if(PHOTONTRACER_BUILD_EMBREE)
  target_link_libraries(photontracer_bindings PRIVATE PhotonTracerEmbree)
endif()
