#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include "run_lapack.h"

//REF: https://scicomp.stackexchange.com/questions/26395/how-to-start-using-lapack-in-c
// dgeev_ is a symbol in the LAPACK library files
// g++ dgesv_stl.cpp  -L/home/hpjeon/sw_local/lapack/3.9.0/lib -llapack
extern "C" {
    extern int dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);
}

void CPU_solver::run_lapack(const int &nn, const int &method){
    int n=nn,m;
    int nrhs, lda, ldb, info;
    std::vector<int> ipiv;
    std::vector<double> A, b;
    m = n;
    nrhs = 1;
    lda = std::max(1,n);
    ldb = lda;
    A_.resize(n*m);
    b_.resize(n);
    ipiv.resize(n);
    if (method == 0) {
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0,1);
	for (int i=0;i<n;i++){
	    for (int j=0;j<m;j++){
		A_[j*n+i] = distribution(generator); 
	    }
	    b_[i] = distribution(generator);
	}
    } else {
	for (int i=0;i<n;i++){
	    for (int j=0;j<m;j++){
		A_[j*n+i] = (double) (n - abs(i-j));
	    }
	    b_[i] = (double) abs(n - 2*i);
	}
    }
    // calculate eigenvalues using the DGEEV subroutine
    dgesv_(&n, &nrhs, A_.data(), &lda, ipiv.data(), b_.data(), &ldb, &info);
    // check for errors
    if (info!=0){
	std::cout << "Error: dgesv returned error code " << info << std::endl;
    }
}

void CPU_solver::deliver_result(std::vector<double> &x){
    x = b_;
}
