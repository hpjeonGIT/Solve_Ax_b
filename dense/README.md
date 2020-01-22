# Code Project
- Dense Matrix Solver

## Objectives
- Solve Ax = b
- Using LAPACK for CPU
- Using CuSolver for GPU
- TBD: Compare performance - wall time, memory foot-print on each
- Unit-tests for each src folder

## Steps to use
- `cd dense`
- `mkdir build; cd build`
- `cmake ..`
- `make -j 3`
- `make install`
- `./bin/dense_solver` # Running the solver executable
- `ctest` # For unit-test

## Code structure
```
└── src
     ├── CPU
     │   └── test
     ├── GPU
     │   └── test
     └── main
         └── test
```
- Code building
  - `mkdir build; cd build; cmake .. ; make -j 4; make install`
  - Link to Google test is necessary. Adjust CMakeLists.txt for the location of libraries
  - May download CuBLAS and LAPACK
  - Will use C++11 features
  - nvcc and gcc with v> 4.6 will be necessary

## Problem configuration
- Matrix A
  - Dense/symmetric matrix
  - Positive definiteness not gauranteed
  - A[i,j] = N-abs(i-j)
- Vector b
  - [1,2,3, ..., N]
- Wil solve Ax=b, printing x vector as results
- N=[3:10], showing wall time cost and memory foot print

## CuSolver sample code
```
//Ref: https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf
//REF: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/cuSolverDn_LinearSolver/cuSolverDn_LinearSolver.cpp
// Command :  nvcc test_cusolverdn.cpp -lcudart -lcublas -lcusolver
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#define BILLION 1000000000L ;

int main(int argc, char** argv) {
    struct timespec start, stop;
    int n, lda;
    n = 10000;
    lda = n;
    double accum ; // elapsed time variable
    cublasStatus_t stat ;
    cudaError cudaStatus ;
    cusolverStatus_t cusolverStatus ;
    cusolverDnHandle_t handle ;
    double *h_A, *h_b; // Host memory. h_b will be the copy of d_x after solve
    double  *d_A, *d_b, * d_Work ; // Device memory, coeff .matrix , rhs , workspace
    int * d_pivot , *d_info , Lwork ; // pivots , info , worksp . size
    int info_gpu = 0;
// prepare memory on the host
    h_A = ( double *) malloc (n*n* sizeof ( double ));
    h_b = ( double *) malloc (n*   sizeof ( double ));
    for (int i=0; i<n ; i++) {
	for (int j=0; j<n; j++) {
	    h_A[i*n + j] = (double) (n - abs(i-j));
	}
	h_b[i] = (double) (n - i*2);
    }
    double al =1.0 , bet =0.0; // coefficients for dgemv
    cudaStatus = cudaGetDevice (0);
    cusolverStatus = cusolverDnCreate (& handle );
    // cusolverDnCreate seems to conflict with thrust::device_memory
    // 0118-2020 
// prepare memory on the device
    cudaStatus = cudaMalloc(( void **)& d_A,     n*n* sizeof (double));
    cudaStatus = cudaMalloc(( void **)& d_b,     n*   sizeof (double));
    cudaStatus = cudaMalloc(( void **)& d_pivot, n*   sizeof (int));
    cudaStatus = cudaMalloc(( void **)& d_info,       sizeof (int ));
    cudaStatus = cudaMemcpy(d_A,h_A,n*n*sizeof(double),cudaMemcpyHostToDevice); 
    cudaStatus = cudaMemcpy(d_b,h_b,n*  sizeof(double),cudaMemcpyHostToDevice);
    //vvvvvvvvvvvvvvvvvvvvvvvv
    cusolverStatus = cusolverDnDgetrf_bufferSize(handle, n, n, d_A, lda, &Lwork ); 
    cudaStatus = cudaMalloc (( void **)& d_Work , Lwork * sizeof (double));
    clock_gettime ( CLOCK_REALTIME ,&start ); // timer start
    cusolverStatus = cusolverDnDgetrf(handle,n,n,d_A,lda,d_Work, d_pivot, d_info);
    cusolverStatus = cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1,
				      d_A, lda, d_pivot, d_b, n, d_info);
    cudaStatus = cudaDeviceSynchronize();
    //^^^^^^^^^^^^^^^^^^
    clock_gettime ( CLOCK_REALTIME ,&stop ); // timer stop
    accum =( stop.tv_sec - start.tv_sec )+ // elapsed time
	( stop.tv_nsec - start.tv_nsec )/( double ) BILLION ;
    printf (" getrf + getrs time : %lf sec .\n",accum ); // print el. time
    cudaStatus = cudaMemcpy (&info_gpu, d_info, sizeof (int),
			     cudaMemcpyDeviceToHost ); // d_info -> info_gpu
    printf (" after getrf + getrs : info_gpu = %d\n", info_gpu );
    cudaStatus = cudaMemcpy (h_b, d_b , n* sizeof (double) ,
			     cudaMemcpyDeviceToHost ); // 
    printf (" solution : ");
    for (int i = 0; i < n; i++) printf ("%g, ", h_b[i]);
    printf (" ... "); // print first components of the sol
    printf ("\n");
// free memory
    cudaStatus = cudaFree (d_A );
    cudaStatus = cudaFree (d_b);

    cudaStatus = cudaFree ( d_pivot );
    cudaStatus = cudaFree ( d_info );
    cudaStatus = cudaFree ( d_Work );
    free (h_A); free (h_b); 
    cusolverStatus = cusolverDnDestroy ( handle );
    cudaStatus = cudaDeviceReset ();
    return 0;
}
```
- Using thrust library didn't work - cusolverDnCreate() seems to conflict with memory allocation of thrust::device_vector

## LAPACK sample code
- Ref: https://scicomp.stackexchange.com/questions/26395/how-to-start-using-lapack-in-c
- Using dgesv

```
#include <iostream>
#include <algorithm>
#include <vector>

//REF: https://scicomp.stackexchange.com/questions/26395/how-to-start-using-lapack-in-c
// dgeev_ is a symbol in the LAPACK library files
extern "C" {
    extern int dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);
}

int main(int argc, char** argv){

  int n,m;
  int nrhs, lda, ldb, info;
  std::vector<int> ipiv;
  std::vector<double> A, b;
  n = 4;
  m = n;
  nrhs = 1;
  lda = std::max(1,n);
  ldb = lda;
  // matrix A
  A.resize(n*m);
  ipiv.resize(n);
  std::cout << "Matrix A=" << std::endl;
  for (int i=0;i<n;i++){
    for (int j=0;j<m;j++){
	A[j*n+i] = (double) (n - std::abs(i-j));
	std::cout << A[j*n+i] << ' ' ;
    }
    std::cout << std::endl;
  }
  //vector b
  b.resize(n);
  std::cout << "Vector b = " ;
  for (std::size_t i=0;i<n; i++) {
      b[i] = (double) abs(n-i*2);
      std::cout << b[i] << ' ' ;
  }
  std::cout << std::endl;
  
  // calculate eigenvalues using the DGEEV subroutine
  dgesv_(&n, &nrhs, A.data(), &lda, ipiv.data(), b.data(), &ldb, &info);
  // check for errors
  if (info!=0){
      std::cout << "Error: dgesv returned error code " << info << std::endl;
    return -1;
  }

  // output eigenvalues to stdout
  std::cout << "Answer x= ";
  for (std::size_t i=0;i<n;i++){
      std::cout << b[i] << ' ';
  }
  std::cout << std::endl;

  return 0;
}
```
- `g++ dgesv_stl.cpp  -L/home/hpjeon/sw_local/lapack/3.9.0/lib -llapack`
