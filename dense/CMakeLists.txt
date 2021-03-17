project(denseSolver CXX CUDA)
cmake_minimum_required(VERSION 2.6)

add_definitions(-std=c++11)
find_package(CUDA REQUIRED)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_VERBOSE_MAKEFILE ON)
set(LAPACK_ROOT "/home/hpjeon/sw_local/lapack/3.9.0")
set(CUDA_ROOT "/usr/local/cuda")
set(CMAKE_CXX_FLAGS "${CMAXE_CXX_FLAGS} -Wall")
#enable_testing()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAXE_CXX_FLAGS} -g -O0")  
else()
  set(CMAKE_CXX_FLAGS "${CMAXE_CXX_FLAGS} -O2")  
endif()
#
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
# Timing for build - Ref: https://stackoverflow.com/questions/5962285/cmake-compilation-statistics
# Per object or target, Elapsed time is printed
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )
include_directories(${CUDA_INCLUDE_DIRS})
set(ALL_CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})
set(LIBS ${LIBS} ${ALL_CUDA_LIBS})
set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
#add_subdirectory(main)
add_subdirectory(src/cpu)
add_subdirectory(src/gpu)
add_subdirectory(src/main)
# Ref: https://stackoverflow.com/questions/6352123/multiple-directories-under-cmake
#https://gitlab.com/CLIUtils/modern-cmake/tree/master/examples/extended-project

