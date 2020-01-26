#include <iostream>
#include <vector>
#include <list>
#include <vector>
#include <string>
#include <mpi.h>
#include "reader.h"

int main(int argc, char** argv) {
    int myid, num_procs;
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    mtrx_reader parsor;
    mtrx_csr spdata;
    std::list<std::string> file_list = {"685_bus.mtx"}; //, "1138_bus.mtx"};
    for (auto fname : file_list){
        std::cout << "reading " << fname << std::endl;
        parsor.from_mtx(fname, spdata, myid, num_procs);
    }
    std::cout << "myid =" << myid << " nnz = " << spdata.nnz_ << std::endl;
    MPI_Finalize();
    return 0;
}
