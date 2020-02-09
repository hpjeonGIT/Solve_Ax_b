include_directories(${GTEST_HOME}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${GTEST_HOME}/lib)
#
add_executable(test_lapack_simple   test_lapack_simple.cpp)
add_executable(test_lapack_residual test_lapack_residual.cpp)
target_link_libraries(test_lapack_simple   run_lapack gtest pthread -L${LAPACK_ROOT}/lib -llapack)
target_link_libraries(test_lapack_residual run_lapack gtest pthread -L${LAPACK_ROOT}/lib -llapack)
#
add_test(NAME lapack_test_simple COMMAND test_lapack_simple)
add_test(NAME lapack_test_residual COMMAND test_lapack_residual)
