#include <iostream>
#include <vector>
#include "cpu/run_lapack.h"
#include "gpu/run_cusolver.h"
#include "dataset.h"

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
