//Ref: https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf
//REF: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/cuSolverDn_LinearSolver/cuSolverDn_LinearSolver.cpp
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#define BILLION 1000000000L ;
#include "run_cusolver.h"
#include "helper_cuda.h"


void GPU_solver::run_cuda_gensolver(const int &nn, const std::vector<double> &Aex,
		const std::vector<double> &bex) {
    struct timespec start, stop;
    int n=nn, lda;
    lda = n;
    double accum ; // elapsed time variable
    cublasStatus_t stat ;
    cudaError cudaStatus ;
    cusolverStatus_t cusolverStatus ;
    cusolverDnHandle_t handle ;
    double *d_A, *d_b, *d_Work; // Device memory, coeff .matrix, rhs, workspace
    int *d_pivot, *d_info, Lwork; // pivots, info, worksp. size
    int info_gpu = 0;
    float free_m,total_m,used_m;
    size_t free_t,total_t;

    // prepare memory on the host
    h_A_.resize(n*n);
    h_b_.resize(n);
    h_A_ = Aex;
    h_b_ = bex;
    cudaStatus = cudaGetDevice(0);
    //checkCudaErrors(cudaStatus); // this yields an error ?
    cusolverStatus = cusolverDnCreate (&handle );
    // cusolverDnCreate seems to conflict with thrust::device_memory
    // 0118-2020 
    // prepare memory on the device
    checkCudaErrors(cudaMalloc(( void **)&d_A,     n*n* sizeof (double)));
    checkCudaErrors(cudaMalloc(( void **)&d_b,     n*   sizeof (double)));
    checkCudaErrors(cudaMalloc(( void **)&d_pivot, n*   sizeof (int)));
    checkCudaErrors(cudaMalloc(( void **)&d_info,       sizeof (int )));
    checkCudaErrors(cudaMemcpy(d_A,h_A_.data(),n*n*sizeof(double),
                    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b,h_b_.data(),n*  sizeof(double),
                    cudaMemcpyHostToDevice));
    //vvvvvvvvvvvvvvvvvvvvvvvv
    checkCudaErrors(cusolverDnDgetrf_bufferSize(handle,n,n,d_A,lda,&Lwork)); 
    checkCudaErrors(cudaMalloc (( void **)&d_Work , Lwork * sizeof (double)));
    clock_gettime ( CLOCK_REALTIME ,&start ); // timer start
    checkCudaErrors(cusolverDnDgetrf(handle,n,n,d_A,lda,d_Work,d_pivot,d_info));
    checkCudaErrors(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, d_A, lda,
                    d_pivot, d_b, n, d_info));
    checkCudaErrors(cudaDeviceSynchronize());
    //^^^^^^^^^^^^^^^^^^
    clock_gettime ( CLOCK_REALTIME ,&stop ); // timer stop
    accum =( stop.tv_sec - start.tv_sec )+ // elapsed time
	( stop.tv_nsec - start.tv_nsec )/( double ) BILLION ;
    printf (" getrf + getrs time : %lf sec .\n",accum ); // print el. time
    checkCudaErrors(cudaMemcpy (&info_gpu, d_info, sizeof (int),
                    cudaMemcpyDeviceToHost )); // d_info -> info_gpu
    //printf (" after getrf + getrs : info_gpu = %d\n", info_gpu );
    checkCudaErrors(cudaMemcpy (h_b_.data(), d_b , n* sizeof (double) ,
                    cudaMemcpyDeviceToHost)); //
    cudaMemGetInfo(&free_t,&total_t);
    free_m =(uint)free_t/1024./1024. ;
    total_m=(uint)total_t/1024./1024.;
    used_m=total_m-free_m;
    std::cout << "Used GPU mem=" << used_m << " MB. Free GPU mem=" << free_m << " MB\n";
    // free memory
    checkCudaErrors(cudaFree (d_A));
    checkCudaErrors(cudaFree (d_b));
    checkCudaErrors(cudaFree (d_pivot));
    checkCudaErrors(cudaFree (d_info));
    checkCudaErrors(cudaFree (d_Work));
    cusolverStatus = cusolverDnDestroy (handle);
    cudaStatus = cudaDeviceReset ();
    checkCudaErrors(cudaStatus);
}

void GPU_solver::run_cuda_symsolver(const int &nn, const std::vector<double> &Aex,
        const std::vector<double> &bex) {
    struct timespec start, stop;
    int n=nn, lda;
    lda = n;
    double accum ; // elapsed time variable
    cublasStatus_t stat ;
    cudaError cudaStatus ;
    cusolverStatus_t cusolverStatus ;
    cusolverDnHandle_t handle ;
    double *d_A, *d_b, *d_Work; // Device memory, coeff .matrix, rhs, workspace
    int *d_pivot, *d_info, Lwork; // pivots, info, worksp. size
    int info_gpu = 0;
    float free_m,total_m,used_m;
    size_t free_t,total_t;
    // prepare memory on the host
    h_A_.resize(n*n);
    h_b_.resize(n);
    h_A_ = Aex;
    h_b_ = bex;
    cudaStatus = cudaGetDevice(0);
    //checkCudaErrors(cudaStatus); // this yields an error ?
    cusolverStatus = cusolverDnCreate (&handle );
    // cusolverDnCreate seems to conflict with thrust::device_memory
    // 0118-2020
    // prepare memory on the device
    checkCudaErrors(cudaMalloc(( void **)&d_A,     n*n* sizeof (double)));
    checkCudaErrors(cudaMalloc(( void **)&d_b,     n*   sizeof (double)));
    checkCudaErrors(cudaMalloc(( void **)&d_pivot, n*   sizeof (int)));
    checkCudaErrors(cudaMalloc(( void **)&d_info,       sizeof (int )));
    checkCudaErrors(cudaMemcpy(d_A,h_A_.data(),n*n*sizeof(double),
                    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b,h_b_.data(),n*  sizeof(double),
                    cudaMemcpyHostToDevice));
    //vvvvvvvvvvvvvvvvvvvvvvvv
    checkCudaErrors(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER,
            n, d_A, lda, &Lwork));
    checkCudaErrors(cudaMalloc (( void **)&d_Work , Lwork * sizeof (double)));
    clock_gettime ( CLOCK_REALTIME ,&start ); // timer start
    checkCudaErrors(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, d_A,
            lda, d_Work, Lwork, d_info));
    checkCudaErrors(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_LOWER, n, 1,
            d_A, lda, d_b, lda, d_info));
    checkCudaErrors(cudaDeviceSynchronize());
    //^^^^^^^^^^^^^^^^^^
    clock_gettime ( CLOCK_REALTIME ,&stop ); // timer stop
    accum =( stop.tv_sec - start.tv_sec )+ // elapsed time
    ( stop.tv_nsec - start.tv_nsec )/( double ) BILLION ;
    printf (" getrf + getrs time : %lf sec .\n",accum ); // print el. time
    checkCudaErrors(cudaMemcpy (&info_gpu, d_info, sizeof (int),
                    cudaMemcpyDeviceToHost )); // d_info -> info_gpu
    //printf (" after getrf + getrs : info_gpu = %d\n", info_gpu );
    checkCudaErrors(cudaMemcpy (h_b_.data(), d_b , n* sizeof (double) ,
                    cudaMemcpyDeviceToHost)); //
    cudaMemGetInfo(&free_t,&total_t);
    free_m =(uint)free_t/1024./1024. ;
    total_m=(uint)total_t/1024./1024.;
    used_m=total_m-free_m;
    std::cout << "Used GPU mem=" << used_m << " MB. Free GPU mem=" << free_m << " MB\n";
    // free memory
    checkCudaErrors(cudaFree (d_A));
    checkCudaErrors(cudaFree (d_b));
    checkCudaErrors(cudaFree (d_pivot));
    checkCudaErrors(cudaFree (d_info));
    checkCudaErrors(cudaFree (d_Work));
    cusolverStatus = cusolverDnDestroy (handle);
    cudaStatus = cudaDeviceReset ();
    checkCudaErrors(cudaStatus);
}

void GPU_solver::deliver_result(std::vector<double> &x){
    x = h_b_;
}
