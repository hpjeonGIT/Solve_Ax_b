#include <iostream>
#include <vector>
#include "cpu/run_lapack.h"

int main(int argc, char** argv) {
    std::vector<int> test_list = {2000, 4000, 8000};
    std::vector<double> x;
    CPU_solver csolver;
    int method;
    // method = 0; random, method=1, symm using 1..N
    for (auto n : test_list) {
	// call CPU LAPACK
	method = 0;
	csolver.run_lapack(n, method);
	x.resize(n);
	csolver.deliver_result(x);
	// call GPU CuSolve
	//run_gpu_cusolver(&n);
	// print wall time
	std::cout << x[0] << "  jeonb " << std::endl;
    }    
    return 0;
}
