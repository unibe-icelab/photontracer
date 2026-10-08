
# Add test executable
add_executable(runTests 
  test_photonTracer.cu)

target_link_libraries(runTests 
  ${GTEST_LIBRARIES}
  ${CUDA_LIBRARIES}
  OptiXToolkit::Memory
  OptiXToolkit::ShaderUtil)

# Set properties for CUDA compilation
set_target_properties(runTests PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME runTests COMMAND runTests)