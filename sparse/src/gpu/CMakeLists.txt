set(gpu_source run_amgx.cpp run_amgx.h)
add_library(gpu_lib SHARED ${gpu_source})
target_link_libraries(gpu_lib -L${CUDA_ROOT}/lib64 -lcudart -L${AMGX_ROOT}/lib -lamgxsh)
include_directories(${AMGX_ROOT}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
# For unit-test
add_subdirectory(test)
