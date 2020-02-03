set(reader_source reader.cpp reader.h)
add_library(reader_lib SHARED ${reader_source})
include_directories(${CMAKE_SOURCE_DIR}/src)
#
set(main_source main.cpp)
add_executable(sparse_solver ${main_source})
target_link_libraries(sparse_solver reader_lib cpu_lib gpu_lib)
install(TARGETS sparse_solver DESTINATION ${CMAKE_BINARY_DIR}/bin)
# 
# For unit-test
add_subdirectory(test)

