set(data_source dataset.cpp dataset.h)
add_library(data_lib SHARED ${data_source})
#
set(main_source main.cpp)
add_executable(dense_solver ${main_source})
target_link_libraries(dense_solver data_lib run_cusolver run_lapack -L${LAPACK_ROOT}/lib -llapack -L${CUDA_ROOT}/lib64 -lcudart -lcublas -lcusolver)
include_directories(${CMAKE_SOURCE_DIR}/src)
install(TARGETS dense_solver DESTINATION ${CMAKE_BINARY_DIR}/bin)
# 
# For unit-test
add_subdirectory(test)

