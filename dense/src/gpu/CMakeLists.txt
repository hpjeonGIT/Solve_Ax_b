set(gpu_source run_cusolver.cpp)
cuda_add_library(run_cusolver SHARED ${gpu_source})
#set_target_properties(run_cusolver PROPERTIES COMPILE_FLAGS "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA")
set_target_properties(run_cusolver PROPERTIES CUDA_SEPERABLE_COMPILATION ON)


