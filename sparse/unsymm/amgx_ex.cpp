#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <mpi.h>
#include "cuda_runtime.h"
#include "amgx_c.h"

/*
export LD_LIBRARY_PATH+=:/home/hpjeon/sw_local/amgx/lib
export AMGX_ROOT=/home/hpjeon/sw_local/amgx
mpicxx -std=c++11 amgx_ex.cpp -I/usr/local/cuda/include -I${AMGX_ROOT}/include \
-L/usr/local/cuda/lib64 -lcudart -L${AMGX_ROOT}/lib -lamgxsh
mpirun -np 3 ./a.out
*/
// The base structure is from amgx_mpi_poisson7.c of AMGX source repo

/* CUDA error macro */
#define CUDA_SAFE_CALL(call) do {                                 \
  cudaError_t err = call;                                         \
  if(cudaSuccess != err) {                                        \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString( err) );       \
    exit(EXIT_FAILURE);                                           \
  } } while (0)


int main(int argc, char **argv)
{
    //MPI (with CUDA GPUs)
    int myid = 0;
    int gpu_id = 0;
    int ncpus = 0;
    int n_global, local_size;
    int gpu_count = 0;
    int nnz;
    std::vector<double> h_x, h_b, data;
    std::vector<int> partition_vector, row_ptr;
    std::vector<long> col_ind_global;

    MPI_Comm amgx_mpi_comm = MPI_COMM_WORLD;
    AMGX_Mode mode;
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    //status handling
    AMGX_SOLVE_STATUS status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(amgx_mpi_comm, &ncpus);
    MPI_Comm_rank(amgx_mpi_comm, &myid);
    if (ncpus != 3) {
        std::cout << "This is for 3 mpi ranks only. Stops here \n";
        MPI_Finalize();
        return 0;
    }
    //CUDA GPUs
    CUDA_SAFE_CALL(cudaGetDeviceCount(&gpu_count));
    gpu_id = myid % gpu_count;
    CUDA_SAFE_CALL(cudaSetDevice(gpu_id));
    printf("Process %d selecting device %d\n", myid, gpu_id);
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    /* system */
    //AMGX_SAFE_CALL(AMGX_install_signal_handler());
    mode = AMGX_mode_dDDI; // precision configuration
    AMGX_SAFE_CALL(AMGX_config_create(&cfg,
               "config_version=2, solver=FGMRES, gmres_n_restart=20,max_iters=100,"
               "norm=L2, convergence=RELATIVE_INI_CORE, monitor_residual=1,"
               "tolerance=1e-4, preconditioner(amg_solver)=AMG,"
               "amg_solver:algorithm=CLASSICAL, amg_solver:max_iters=2,"
               "amg_solver:presweeps=1, amg_solver:postsweeps=1, amg_solver:cycle=V,"
               "print_solve_stats=1, print_grid_stats=1, obtain_timings=1,"
               "exception_handling=1"));
    /* create resources, matrix, vector and solver */
    AMGX_resources_create(&rsrc, cfg, &amgx_mpi_comm, 1, &gpu_id);
    AMGX_matrix_create(&A, rsrc, mode);
    n_global = 10;
    partition_vector = {0,0,0,0,  1,1,1, 2,2,2};
    if (myid == 0){
     /*    10    8    0    0    0    0    0    0    0    0
            7   10    8    0    0    0    0    0    0    0
            0    7   10    8    0    0    0    0    0    0
            0    0    7   10    8    0    0    0    0    0 */
        local_size = 4;
        data = {10., 8., 7., 10., 8., 7., 10., 8., 7., 10., 8.};
        col_ind_global = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4};
        row_ptr = {0, 2, 5, 8, 11};
        nnz = 11;
        AMGX_matrix_upload_all_global(A, n_global, local_size, nnz, 1,1, &row_ptr[0],
                &col_ind_global[0], &data[0], NULL, 2, 2, &partition_vector[0]);
        h_x = {0, 0, 0, 0};
        h_b = {1., 2., 3., 4.};
    } else if (myid == 1) {
    /*  0    0    0    7   10    8    0    0    0    0
        0    0    0    0    7   10    8    0    0    0
        0    0    0    0    0    7   10    8    0    0 */
        local_size = 3;
        data = {7., 10., 8., 7., 10., 8., 7., 10., 8.};
        col_ind_global = {3, 4, 5, 4, 5, 6, 5, 6, 7};
        row_ptr = {0, 3, 6, 9};
        nnz = 9;
        AMGX_matrix_upload_all_global(A, n_global, local_size, nnz, 1,1, &row_ptr[0],
                &col_ind_global[0], &data[0], NULL, 2, 2, &partition_vector[0]);
        h_x = {0, 0, 0};
        h_b = {5., 6., 7.};
    } else {// myid == 2
   /*   0    0    0    0    0    0    7   10    8    0
        0    0    0    0    0    0    0    7   10    8
        0    0    0    0    0    0    0    0    7   10 */
        local_size = 3;
        data = {7., 10., 8., 7., 10., 8., 7., 10};
        col_ind_global = {6, 7, 8, 7, 8, 9, 8, 9};
        row_ptr = {0, 3, 6, 8};
        nnz = 8;
        AMGX_matrix_upload_all_global(A, n_global, local_size, nnz, 1,1, &row_ptr[0],
                &col_ind_global[0], &data[0], NULL, 2, 2, &partition_vector[0]);
        h_x = {0, 0, 0};
        h_b = {8., 9., 10.};
    }
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);
    AMGX_solver_create(&solver, rsrc, mode, cfg);
    int nrings = 2; // 1 ring for aggregation and 2 rings for classical path
    AMGX_config_get_default_number_of_rings(cfg, &nrings);
    // data
    AMGX_vector_bind(x, A);
    AMGX_vector_bind(b, A);
    /* upload the vector (and the connectivity information) */
    AMGX_vector_upload(x, local_size, 1, &h_x[0]);
    AMGX_vector_upload(b, local_size, 1, &h_b[0]);
    AMGX_solver_setup(solver, A);
    AMGX_solver_solve(solver, b, x);
    AMGX_solver_get_status(solver, &status);
    AMGX_vector_download(x, &h_x[0]);
    // print result_host
    std::cout << "myid = " << myid << " x=";
    for (int i=0; i < local_size; i++)
            std::cout << " " << h_x[i];
    std::cout << std::endl;
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg))
    AMGX_SAFE_CALL(AMGX_finalize_plugins())
    AMGX_SAFE_CALL(AMGX_finalize())
    MPI_Finalize();
    CUDA_SAFE_CALL(cudaDeviceReset());
   return status;
}
