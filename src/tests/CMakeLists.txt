# GPU tests: the physics functions run in CUDA kernels
if(PHOTONTRACER_BUILD_OPTIX)
  add_executable(runTests
    test_light_scattering.cu)

  target_link_libraries(runTests
    ${GTEST_LIBRARIES}
    ${CUDA_LIBRARIES}
    OptiXToolkit::Memory
    OptiXToolkit::ShaderUtil)

  set_target_properties(runTests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  add_test(NAME runTests COMMAND runTests)
endif()

# CPU tests: the same functions built as plain C++ without CUDA headers
add_executable(runHostTests
  test_light_scattering_host.cpp
  test_material_hit.cpp
  test_pcg32.cpp
  test_ray_generation.cpp)

target_compile_definitions(runHostTests PRIVATE PHOTONTRACER_NO_CUDA)
target_link_libraries(runHostTests ${GTEST_LIBRARIES})

add_test(NAME runHostTests COMMAND runHostTests)

# Embree tests: the spawn points are checked against Embree's own intersector
if(PHOTONTRACER_BUILD_EMBREE)
  add_executable(runEmbreeTests test_spawn_point.cpp)
  target_compile_definitions(runEmbreeTests PRIVATE PHOTONTRACER_NO_CUDA)
  target_include_directories(runEmbreeTests PRIVATE ${CMAKE_SOURCE_DIR}/src/photonTracer)
  target_link_libraries(runEmbreeTests GTest::gtest_main embree TBB::tbb)

  add_test(NAME runEmbreeTests COMMAND runEmbreeTests)
endif()
