#include <iostream>
#include <algorithm>
#include <vector>
#include <sys/resource.h>
#include <chrono>
#include "run_lapack.h"

//REF: https://scicomp.stackexchange.com/questions/26395/how-to-start-using-lapack-in-c
// dgeev_ is a symbol in the LAPACK library files
// g++ dgesv_stl.cpp  -L/home/hpjeon/sw_local/lapack/3.9.0/lib -llapack
extern "C" {
    extern int dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);
}
extern "C" {
    extern int dsysv_(char*, int*, int*, double*, int*, int*, double*, int*,
            double*, int*, int*);
}

void CPU_solver::run_gensolver(const int &nn, const std::vector<double> &Aex,
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
    struct rusage usage;
    auto start = std::chrono::system_clock::now();
    getrusage(RUSAGE_SELF, &usage);
    dgesv_(&n, &nrhs, A_.data(), &lda, ipiv.data(), b_.data(), &ldb, &info);
    getrusage(RUSAGE_SELF, &usage);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Allocated mem " << usage.ru_maxrss/1024 << " MB\n";
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "sec\n";
    // check for errors
    if (info!=0){
        std::cout << "Error: dgesv returned error code " << info << std::endl;
    }
}

void CPU_solver::run_symsolver(const int &nn, const std::vector<double> &Aex,
        const std::vector<double> &bex){
    int n=nn,m;
    int nrhs, lda, ldb, info, lwork;
    double wkopt;
    std::vector<double> work;
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
    struct rusage usage;
    auto start = std::chrono::system_clock::now();
    getrusage(RUSAGE_SELF, &usage);
    lwork = -1;
    dsysv_("Lower", &n, &nrhs, A_.data(), &lda, ipiv.data(), b_.data(),
            &ldb, &wkopt, &lwork, &info);
    lwork = static_cast<int> (wkopt);
    work.resize(lwork);
    dsysv_("Lower", &n, &nrhs, A_.data(), &lda, ipiv.data(), b_.data(),
                &ldb, work.data(), &lwork, &info);
    getrusage(RUSAGE_SELF, &usage);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Allocated mem " << usage.ru_maxrss/1024 << " MB\n";
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "sec\n";
    // check for errors
    if (info!=0){
        std::cout << "Error: dgesv returned error code " << info << std::endl;
    }
}

void CPU_solver::deliver_result(std::vector<double> &x){
    x = b_;
}
