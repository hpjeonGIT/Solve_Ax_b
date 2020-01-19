set(cpu_source run_lapack.cpp run_lapack.h)
add_library(run_lapack SHARED ${cpu_source})
