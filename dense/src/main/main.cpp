#include <iostream>
#include <vector>
#include <random>
#include "cpu/run_lapack.h"
#include "gpu/run_cusolver.h"
#include "dataset.h"

void Dataset::random_matrix(const int &n){
	A_.resize(n*n);
	b_.resize(n);
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0,1);
	for (int i=0;i<n;i++){
	    for (int j=0;j<n;j++){
		A_[j*n+i] = distribution(generator);
	    }
	    b_[i] = distribution(generator);
	}
}

void Dataset::sym_matrix(const int &n) {
	A_.resize(n*n);
	b_.resize(n);
	for (int i=0;i<n;i++){
	    for (int j=0;j<n;j++){
		A_[j*n+i] = (double) (n - abs(i-j));
	    }
	    b_[i] = (double) abs(n - 2*i);
	}
}

std::vector<double> Dataset::return_A() {
	return A_;
}

std::vector<double> Dataset::return_b() {
	return b_;
}

int main(int argc, char** argv) {
    std::vector<int> test_list = {2000, 4000, 8000};
    std::vector<double> x;
    Dataset q;
    CPU_solver csolver;
    GPU_solver gsolver;
    int method = 0;
    // method = 0; random, method=1, symm using 1..N
    for (auto n : test_list) {
    	// make sample data
    	if (method == 0){
    		q.random_matrix(n);
    	}
		else {
			q.sym_matrix(n);
		}
		// call CPU LAPACK
    	csolver.run_lapack(n, q.return_A(), q.return_b());
		x.resize(n);
		csolver.deliver_result(x);
		std::cout << x[0] << " lapack result" << std::endl;
		// call GPU CuSolve
		gsolver.run_cusolver(n, q.return_A(), q.return_b());
		gsolver.deliver_result(x);
		// print wall time
		std::cout << x[0] << "  cusolver result" << std::endl;
	}
    return 0;
}
