#include <iostream>
#include <algorithm>
#include <vector>
#include "run_lapack.h"

//REF: https://scicomp.stackexchange.com/questions/26395/how-to-start-using-lapack-in-c
// dgeev_ is a symbol in the LAPACK library files
// g++ dgesv_stl.cpp  -L/home/hpjeon/sw_local/lapack/3.9.0/lib -llapack
extern "C" {
    extern int dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);
}

void CPU_solver::run_lapack(const int &nn, const std::vector<double> &Aex,
		const std::vector<double> &bex){
    int n=nn,m;
    int nrhs, lda, ldb, info;
    std::vector<int> ipiv;
    m = n;
    nrhs = 1;
    lda = std::max(1,n);
    ldb = lda;
    A_.resize(n*m);
    b_.resize(n);
    ipiv.resize(n);
    A_ = Aex;
    b_ = bex;
    //std::cout << "b0= " << b_[0] << std::endl;
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
