set(reader_source reader.cpp reader.h)
add_library(reader_lib SHARED ${reader_source})
#
set(main_source main.cpp)
add_executable(sparse_solver ${main_source})
target_link_libraries(sparse_solver reader_lib)
# run_cusolver run_lapack -L${LAPACK_ROOT}/lib -llapack 
#                     -L${CUDA_ROOT}/lib64 -lcudart -lcublas -lcusolver)
include_directories(${CMAKE_SOURCE_DIR}/src)
install(TARGETS sparse_solver DESTINATION ${CMAKE_BINARY_DIR}/bin)
# 
# For unit-test
#add_subdirectory(test)

