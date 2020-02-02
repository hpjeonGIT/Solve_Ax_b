#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <mpi.h>
#include "run_amgx.h"
#include "cuda_runtime.h"
#include "amgx_c.h"

/* CUDA error macro */
#define CUDA_SAFE_CALL(call) do {                                 \
  cudaError_t err = call;                                         \
  if(cudaSuccess != err) {                                        \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString( err) );       \
    exit(EXIT_FAILURE);                                           \
  } } while (0)


void AMGX_solver::run_amgx(mtrx_csr &spdata, rhs &b_v, int const &myid,
        int const &num_procs) {

    int gpu_id = 0;
    int gpu_count = 0;
    std::vector<double> h_x, h_b, data;
    std::vector<int> row_ptr;
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
    //CUDA GPUs
    CUDA_SAFE_CALL(cudaGetDeviceCount(&gpu_count));
    gpu_id = myid % gpu_count;
    CUDA_SAFE_CALL(cudaSetDevice(gpu_id));
    printf("Process %d selecting device %d\n", myid, gpu_id);
    std::vector<int> local_vec(num_procs);
    MPI_Allgather(&spdata.local_size_, 1, MPI_INT,local_vec.data(), 1,
            MPI_INT, MPI_COMM_WORLD);
    std::vector<int> partition_vector(spdata.global_size_);
    int niter = 0;
    for (int i=0; i< num_procs; i++) {
        for (int j=0; j< local_vec[i]; j++){
            partition_vector[niter] = i;
            niter ++ ;
        }
    }
    if (niter != spdata.global_size_) {
        std::cout << "Something is wrong. Sum of loca_size_ doesn't match global_size \n";
        std::cout << "global size = " << spdata.global_size_ << std::endl;
        std::cout << "counted number = " << niter << std::endl;
        throw;
    }
    for (int i=0; i< spdata.row_ptr_.size(); i++ ) {
    std::cout << " rows_ " << spdata.row_ptr_[i] ;
    }
    for (int i=0; i< spdata.g_colidx_.size(); i++ ) {
    std::cout << " coldix_ " << spdata.g_colidx_[i] ;
    }
    for (int i=0; i< spdata.nnz_v_.size(); i++ ) {
    std::cout << " nnz_v_ " << spdata.nnz_v_[i] ;
    }
    for (int i=0; i< b_v.rows_.size(); i++ ) {
        std::cout << " b_v.rows_ " << b_v.rows_[i] << " " << b_v.values_[i] << std::endl;
    }
    for (int i=0; i< partition_vector.size(); i++ ) {
        std::cout << myid <<" partition_vector " << partition_vector[i] << std::endl;
    }


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

    AMGX_matrix_upload_all_global(A, spdata.global_size_, spdata.local_size_,
            spdata.nnz_, 1, 1, &spdata.row_ptr_[0], &spdata.g_colidx_[0],
            &spdata.values_[0], NULL, 2, 2, &partition_vector[0]);
    x_values_.resize(spdata.local_size_, 0.0);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);
    AMGX_solver_create(&solver, rsrc, mode, cfg);
    int nrings = 2; // 1 ring for aggregation and 2 rings for classical path
    AMGX_config_get_default_number_of_rings(cfg, &nrings);
    // data
    AMGX_vector_bind(x, A);
    AMGX_vector_bind(b, A);
    /* upload the vector (and the connectivity information) */
    AMGX_vector_upload(x, spdata.local_size_, 1, &x_values_[0]);
    AMGX_vector_upload(b, spdata.local_size_, 1, &b_v.values_[0]);
    AMGX_solver_setup(solver, A);
    AMGX_solver_solve(solver, b, x);
    AMGX_solver_get_status(solver, &status);
    AMGX_vector_download(x, &x_values_[0]);
    // print result_host
    std::cout << "myid = " << myid << " x=";
    for (int i=0; i < spdata.local_size_; i++)
            std::cout << " " << x_values_[i];
    std::cout << std::endl;
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg))
    AMGX_SAFE_CALL(AMGX_finalize_plugins())
    AMGX_SAFE_CALL(AMGX_finalize())
    CUDA_SAFE_CALL(cudaDeviceReset());
}
