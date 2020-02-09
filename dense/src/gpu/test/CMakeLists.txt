include_directories(${GTEST_HOME}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${GTEST_HOME}/lib)
#
add_executable(test_cusolver_simple   test_cusolver_simple.cpp)
add_executable(test_cusolver_residual test_cusolver_residual.cpp)
target_link_libraries(test_cusolver_simple   run_cusolver gtest pthread -L${CUDA_ROOT}/lib64 -lcudart -lcublas -lcusolver)
target_link_libraries(test_cusolver_residual run_cusolver gtest pthread -L${CUDA_ROOT}/lib64 -lcudart -lcublas -lcusolver)
#
add_test(NAME cusolver_test_simple COMMAND test_cusolver_simple)
add_test(NAME cusolver_test_residual COMMAND test_cusolver_residual)
