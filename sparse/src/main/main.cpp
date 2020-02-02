#include <iostream>
#include <vector>
#include <vector>
#include <string>
#include <mpi.h>
#include "reader.h"
#include "cpu/run_hypre.h"

int main(int argc, char** argv) {
    int myid, num_procs;
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    // data setup
    mtrx_reader parsor;
    mtrx_csr spdata;
    rhs  b_v;
    HYPRE_solver cpusolver;
    //Amgx_solver gpusolver;
    std::vector<double> x;
    std::vector<std::string> file_list = {"simple.mtx"} ;//, "685_bus.mtx"}; //, "1138_bus.mtx"};
    std::vector<bool> symm_list = {true} ;//, true};
    for (int i=0; i <  file_list.size(); i++) {
        auto fname = file_list[i];
        auto isSym = symm_list[i];
        std::cout << "reading " << fname << std::endl;
        parsor.from_mtx(fname, isSym, spdata, myid, num_procs);
        parsor.set_b(spdata, b_v, myid, num_procs);
        cpusolver.run_hypre(spdata, b_v, myid);
        x.resize(spdata.local_size_);
        cpusolver.get_result(x);
        //gpusolver.run_amgx(myid, spdata);
    }
    std::cout << "myid =" << myid << " nnz = " << spdata.nnz_ << std::endl;
    MPI_Finalize();
    return 0;
}
