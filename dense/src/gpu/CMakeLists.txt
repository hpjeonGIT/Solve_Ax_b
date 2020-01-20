set(gpu_source run_cusolver.cu)
cuda_add_library(run_cusolver SHARED ${gpu_source} )
#set_target_properties(run_cusolver PROPERTIES COMPILE_FLAGS "-std=c++11")
set_target_properties(run_cusolver PROPERTIES CUDA_SEPERABLE_COMPILATION ON)


