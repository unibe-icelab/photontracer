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
  test_ray_generation.cpp)

target_compile_definitions(runHostTests PRIVATE PHOTONTRACER_NO_CUDA)
target_link_libraries(runHostTests ${GTEST_LIBRARIES})

add_test(NAME runHostTests COMMAND runHostTests)
