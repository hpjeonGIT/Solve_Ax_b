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


void GPU_solver::run_cusolver(const int &nn, const int &method) {
    struct timespec start, stop;
    int n=nn, lda;
    lda = n;
    double accum ; // elapsed time variable
    cublasStatus_t stat ;
    cudaError cudaStatus ;
    cusolverStatus_t cusolverStatus ;
    cusolverDnHandle_t handle ;
    //double *h_A, *h_b; // Host memory. h_b will be the copy of d_x after solve
    double *d_A, *d_b, *d_Work; // Device memory, coeff .matrix, rhs, workspace
    int *d_pivot, *d_info, Lwork; // pivots, info, worksp. size
    int info_gpu = 0;
// prepare memory on the host
    //h_A = ( double *) malloc (n*n* sizeof ( double ));
    //h_b = ( double *) malloc (n*   sizeof ( double ));
    h_A_.resize(n*n);
    h_b_.resize(n);
    if (method ==0) {
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0,1);
	for (int i=0;i<n;i++){
	    for (int j=0;j<m;j++){
		A_[j*n+i] = distribution(generator); 
	    }
	    b_[i] = distribution(generator);
	}
    } else {
	for (int i=0; i<n ; i++) {
	    for (int j=0; j<n; j++) {
		h_A_[i*n + j] = (double) (n - abs(i-j));
	    }
	    h_b_[i] = (double) (n - i*2);
	}
    }
    cudaStatus = cudaGetDevice (0);
    checkCudaErrors(cudaStatus);
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
    //printf (" getrf + getrs time : %lf sec .\n",accum ); // print el. time
    checkCudaErrors(cudaMemcpy (&info_gpu, d_info, sizeof (int),
				cudaMemcpyDeviceToHost )); // d_info -> info_gpu
    //printf (" after getrf + getrs : info_gpu = %d\n", info_gpu );
    checkCudaErrors(cudaMemcpy (h_b_.data(), d_b , n* sizeof (double) ,
				cudaMemcpyDeviceToHost)); // 
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
