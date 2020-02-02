set(cpu_source run_hypre.cpp run_hypre.h)
add_library(cpu_lib STATIC ${cpu_source})
target_link_libraries(cpu_lib -L${HYPRE_ROOT}/lib -lHYPRE)
include_directories(${HYPRE_ROOT}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
# For unit-test
#add_subdirectory(test)

