cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

# Build options
option(PHOTONTRACER_BUILD_TESTS "Build tests" OFF)
option(ENABLE_DEBUG_PRINT_KERNEL "Enable debug prints in CUDA kernels" OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
endif()

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

project(PlanetaryRaytracer LANGUAGES C CXX CUDA)

if(NOT TARGET CUDA::cuda_driver)
  find_package( CUDAToolkit 12.0 REQUIRED )
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  cmake_language(GET_VERSION_COMPONENT CUDA_MAJOR "${CUDAToolkit_VERSION}" 0)
  if(CUDA_MAJOR GREATER_EQUAL 13)
    set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90-real 100-real 103-real 110-real 120-real 120-virtual)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 50-real 52-real 60-real 61-real 70-real 75-real 80-real 86-real 89-real 90-real 90-virtual)
  endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_CONFIGURATION_TYPES MATCHES "Debug")
  set(IS_DEBUG_BUILD TRUE)
else()
  set(IS_DEBUG_BUILD FALSE)
endif()

if(IS_DEBUG_BUILD)
  message(STATUS "Debug build")
  if(MSVC)
    # MSVC debug flags
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od")
    # NVCC device debug flag is -G; avoid adding Unix -g for host on MSVC
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
  else()
    # GCC/Clang / nvcc on Unix-like
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
  endif()
endif()

if(ENABLE_DEBUG_PRINT_KERNEL)
    add_compile_definitions(DEBUG_PRINT_KERNEL)
endif()

# Add OptiX Toolkit
set(OptiX_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/optix" CACHE PATH "Path to OptiX installation")
set(OTK_BUILD_EXAMPLES OFF CACHE BOOL "Build OTK examples")
set(OTK_BUILD_TESTS OFF CACHE BOOL "Build OTK tests")
set(OTK_BUILD_DOCS OFF CACHE BOOL "Build OTK documentation")
set(OTK_LIBRARIES "ShaderUtil" CACHE STRING "OTK libraries to build")
set(OTK_USE_VCPKG OFF CACHE BOOL "Use vcpkg to find dependenciies")
add_subdirectory(external/otk)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/otk/CMake")

include(FetchOptiX)

if(PHOTONTRACER_BUILD_TESTS)
  find_package( GTest REQUIRED )
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(external/pybind11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)

if(PHOTONTRACER_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(src)