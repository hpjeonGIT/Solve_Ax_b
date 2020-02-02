# Code Project
- Sparse Matrix Solver

## Objectives
- Solve Ax = b
- Using HYPRE for CPU
- Using Amgx for GPU
- TBD: Compare performance - wall time, memory foot-print on each
- Unit-tests for each src folder

## Algebraic multigrid
- Commonly multigrid method is used for structural grids - changing grid resolution along V or W cycles
- Algebraic multigrid enables MG method on purely matrix - without physical grid systems 
- Therefore, even unstructured grids or particle systems can use MG to solve their matrix equations
- As there is no grid, Yair Shapira called this method as the algebraic multilevel method

## Steps to use
- Install HYPRE, AMGX, CUDA, google Test and adjust HYPRE_ROOT, AMGX_ROOT, CUDA_ROOT, GTEST_HOME in the `sparse/CMakeLists.txt`
- `cd sparse`
- `mkdir build; cd build`
- `cmake .. -DCMAKE_CXX_COMPILER=mpicxx`
- `make -j 3`
- `make install`
- `mpirun -n 3 ./bin/sparse_solver` # Running the solver executable
- `ctest` # For unit-test

## Code structure
```
└── src
     ├── CPU
     │   └── test
     ├── GPU
     │   └── test
     └── main
         └── test
```

## Input matrix
- Sample matrices are at sparse/data
- They are matrix market formats from https://sparse.tamu.edu/HB/685_bus and https://www.cise.ufl.edu/research/sparse/matrices/HB/1138_bus.html
- simple4.mtx and simple10.mtx are toy matices, and you may copy to simple.mtx
- Vector b is given arbitrarily
- Wil solve Ax=b, printing x vector as results

## CuSparse sample code
```
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <mpi.h>
#include "cuda_runtime.h"
#include "amgx_c.h"

/*
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
    std::vector<long long int> col_ind_global;

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
            8   10    8    0    0    0    0    0    0    0
            0    8   10    8    0    0    0    0    0    0
            0    0    8   10    8    0    0    0    0    0 */
        local_size = 4;
        data = {10., 8., 8., 10., 8., 8., 10., 8., 8., 10., 8.};
        col_ind_global = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4};
        row_ptr = {0, 2, 5, 8, 11};
        nnz = 11;
        AMGX_matrix_upload_all_global(A, n_global, local_size, nnz, 1,1, &row_ptr[0],
                &col_ind_global[0], &data[0], NULL, 2, 2, &partition_vector[0]);
        h_x = {0, 0, 0, 0};
        h_b = {1., 2., 3., 4.};
    } else if (myid == 1) {
    /*  0    0    0    8   10    8    0    0    0    0
        0    0    0    0    8   10    8    0    0    0
        0    0    0    0    0    8   10    8    0    0 */
        local_size = 3;
        data = {8., 10., 8., 8., 10., 8., 8., 10., 8.};
        col_ind_global = {3, 4, 5, 4, 5, 6, 5, 6, 7};
        row_ptr = {0, 3, 6, 9};
        nnz = 9;
        AMGX_matrix_upload_all_global(A, n_global, local_size, nnz, 1,1, &row_ptr[0],
                &col_ind_global[0], &data[0], NULL, 2, 2, &partition_vector[0]);
        h_x = {0, 0, 0};
        h_b = {5., 6., 7.};
    } else {// myid == 2
   /*   0    0    0    0    0    0    8   10    8    0
        0    0    0    0    0    0    0    8   10    8
        0    0    0    0    0    0    0    0    8   10 */
        local_size = 3;
        data = {8., 10., 8., 8., 10., 8., 8., 10};
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
```

## HYPRE sample code
```
/*
 * Command to run
export HYPRE_ROOT=/home/hpjeon/sw_local/hypre/2.18.2
mpicxx -o a.exe -std=c++11 hypre_ex.cpp -I$HYPRE_ROOT/include -L$HYPRE_ROOT/lib -lHYPRE
mpirun -np 3 ./a.exe
*/
#include <cmath>
#include <iostream>
#include <vector>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

// #include "vis.c"
int main(int argc, char **argv) {

    int myid, num_procs;
    int N, n;
    int ilower, iupper;
    int local_size;
    int solver_id;
    HYPRE_IJMatrix A;
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVector x;
    HYPRE_ParVector par_x;
    HYPRE_Solver solver, precond;
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (num_procs != 3) {
        MPI_Finalize();
        std::cout << "This program runs with 3 ranks ONLY \n";
       return 0;
    }
    /* Default problem parameters */
    N = 10;
    solver_id = 0;
    if (myid ==0) {
        ilower = 0; iupper = 3; local_size = 4;
    } else if (myid == 1) {
        ilower = 4; iupper = 6; local_size = 3;
    }
    else { //myid == 2
        ilower = 7; iupper = 9; local_size = 3;
    }
    local_size = iupper - ilower + 1;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
    // Note that this is for a symmetric matrix, ilower/iupper of row and ilower/iupper of column are same
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);
    {
        std::vector<double> values;
        std::vector<int> cols, nnz_vec, row_vec;
        int nnz, nrows;
        if (myid == 0){
         /*    10    8    0    0    0    0    0    0    0    0
                8   10    8    0    0    0    0    0    0    0
                0    8   10    8    0    0    0    0    0    0
                0    0    8   10    8    0    0    0    0    0 */
        /*    values = {10., 8.}; cols = {0,1}; nnz = 2; n=0;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {8., 10., 8.}; cols = {0,1,2}; nnz = 3; n=1;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {8., 10., 8.}; cols = {1,2,3}; nnz = 3; n=2;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {8., 10., 8.}; cols = {2,3,4}; nnz = 3; n=3;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
        */
            values = {10.,8., 8.,10.,8., 8.,10.,8., 8.,10.,8.};
            cols = {0,1,  0,1,2,  1,2,3,  2,3,4};
            nnz_vec = {2, 3, 3, 3};
            nrows = 4;
            row_vec = {0,1,2,3};
            HYPRE_IJMatrixSetValues(A, nrows, &nnz_vec[0], &row_vec[0], &cols[0], &values[0]);
} else if (myid == 1) {
        /*  0    0    0    8   10    8    0    0    0    0
            0    0    0    0    8   10    8    0    0    0
            0    0    0    0    0    8   10    8    0    0 */
            values = {8., 10., 8.}; cols = {3,4,5}; nnz = 3; n=4;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {8., 10., 8.}; cols = {4,5,6}; nnz = 3; n=5;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {8., 10., 8.}; cols = {5,6,7}; nnz = 3; n=6;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
        } else {// myid == 2
       /*   0    0    0    0    0    0    8   10    8    0
            0    0    0    0    0    0    0    8   10    8
            0    0    0    0    0    0    0    0    8   10 */
            values = {8., 10., 8.}; cols = {6,7,8}; nnz = 3; n=7;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {8., 10., 8.}; cols = {7,8,9}; nnz = 3; n=8;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {8., 10.};      cols = {8,9}; nnz = 2; n=9;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
        }
    }
    HYPRE_IJMatrixAssemble(A);
//       HYPRE_IJMatrixPrint(A, "IJ.out.A");
//       HYPRE_IJVectorPrint(b, "IJ.out.b");
    HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&b);
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);
    // Configuration of RHS
    std::vector<double> rhs_values(local_size), x_values(local_size, 0.0);
    std::vector<int> rows(local_size);
    rhs_values = {};
    // b = [1    2    3    4    5    6    7    8    9   10]
    if (myid == 0) {
        rhs_values = {1., 2., 3., 4.};
        rows = {0, 1, 2, 3};
    } else if (myid == 1) {
        rhs_values = {5., 6., 7.};
        rows = {4, 5, 6};
    } else { // myid ==2
        rhs_values = {8., 9., 10.};
        rows = {7, 8, 9};
    }
    HYPRE_IJVectorSetValues(b, local_size, &rows[0], &rhs_values[0]);
    HYPRE_IJVectorSetValues(x, local_size, &rows[0], &x_values[0]);
    rhs_values.clear();
    HYPRE_IJVectorAssemble(b);
    HYPRE_IJVectorGetObject(b, (void **) &par_b);
    HYPRE_IJVectorAssemble(x);
    HYPRE_IJVectorGetObject(x, (void **) &par_x);
    // FGMRES + AMG preconditioner
    int    num_iterations;
    double final_res_norm;
    int    restart = 30;
    int    modify = 1;
    HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &solver);
    HYPRE_FlexGMRESSetKDim(solver, restart);
    HYPRE_FlexGMRESSetMaxIter(solver, 1000); /* max iterations */
    HYPRE_FlexGMRESSetTol(solver, 1e-7); /* conv. tolerance */
    HYPRE_FlexGMRESSetPrintLevel(solver, 2); /* print solve info */
    HYPRE_FlexGMRESSetLogging(solver, 1); /* needed to get run info later */
    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info */
    HYPRE_BoomerAMGSetCoarsenType(precond, 6);
    HYPRE_BoomerAMGSetOldDefault(precond);
    HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
    HYPRE_BoomerAMGSetNumSweeps(precond, 1);
    HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
    HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */
    HYPRE_FlexGMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                        (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
    HYPRE_ParCSRFlexGMRESSetup(solver, parcsr_A, par_b, par_x);
    HYPRE_ParCSRFlexGMRESSolve(solver, parcsr_A, par_b, par_x);
    HYPRE_FlexGMRESGetNumIterations(solver, &num_iterations);
    HYPRE_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
    if (myid == 0)    {
       printf("\n");
       printf("Iterations = %d\n", num_iterations);
       printf("Final Relative Residual Norm = %e\n", final_res_norm);
       printf("\n");
    }
    HYPRE_ParCSRFlexGMRESDestroy(solver);
    HYPRE_BoomerAMGDestroy(precond);
    HYPRE_IJVectorGetValues(x, local_size, &rows[0], &x_values[0]);
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(b);
    HYPRE_IJVectorDestroy(x);
    std::cout << "myid = " << myid << " x= ";
    for (int i=0; i< local_size;i++) {
        std::cout << x_values[i] << " ";
    }
    std::cout << std::endl;
      /* Finalize MPI*/
    MPI_Finalize();
    return(0);
}
```
- When HYPRE is built, use `./configure --prefix=/home/hpjeon/sw_local/hypre/2.18.2_bigint --enable-shared --enable-bigint` for very large matrix calculations
    - long or 64 bit integer is used for rows_ and colidx_
- For default HYPRE installation, 32bit integer becomes default. Adjust custom_int of src/main/reader.h accordinly    

## Note for cuda libraries
- CuBLAS for dense matrix operation such as matmul. No solver
- CuSparse for sparse matrix operation such as matmul. No solver
- CuSolver for solver. Both of dense and sparse matrices
- MPI support for CuSolver is not yet
- For a distributed-GPU sparse solver, AmgX is the only available library as of Q1-2020 (I guess...)

## Matrix market format
```
1 1 x
2 1 x
3 1 x
2 2 x
3 2 x
...
```
First column is the row, and the second column is the column index. CSR splits the matrix elements along row index, and the appropriate form would be:
```
1 1 x
2 1 x
2 2 x
3 1 x
3 2 x
...
```
In other words, the vector of rows, colidx, values can be split contiguously, yi
elding higher prefetching efficiency.

- In case of of symmetric MM, extra storage must be added. IJ form of HYPRE/Amgx
 doesn't support symmetric matrices, and elements must exist explicitly.

## strategy
- Reads MM data as rows, cols, and val. For a symmetric matrix, (i,j) component 
which i!=j must be duplicated in (j,i).
    - rows: 0 1 2 0 1 2 2 0 1 0 2 ...
    - cols: 0 0 0 1 1 1 2 3 3 4 4 ...
- Then sort cols and val using rows
    - For the cols which have same rows, re-sorted with val vector
- Expectation
    - rows: 0 0 0 0 1 1 1 2 2 2 2 ...
    - cols: 0 1 3 4 0 1 3 0 1 2 4 ...
- Sample code shown below. -std=c++14 is required
```
#include <vector>
#include <algorithm>
#include <iostream>
//ref: https://stackoverflow.com/questions/37368787/c-sort-one-vector-based-on-another-one/46370189

int main(int argc, char** argv) {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> values;
    
    struct csr{
    int row;
    int col;
    double val;       
    };

    rows = {1,2,3,  2,3,4, 3,4,5, 4,5};
    cols = {1,1,1,  2,2,2, 3,3,3, 1,2 };
    values = {1,1,1, 2,2,2, 3,3,3, 9, 10};
    std::vector<csr> abc;
    int n = 11;
    abc.resize(n);
    for (int i=0; i< n; i++) {
    abc[i].row = rows[i];
    abc[i].col = cols[i];
    abc[i].val = values[i];
    }
    std::sort(abc.begin(), abc.end(),
          [](const auto& i, const auto& j) { return i.row < j.row; } );    
    for (int i=0;i<n;i++) std::cout << abc[i].row;
    std::cout << std::endl;
    for (int i=0;i<n;i++) std::cout << abc[i].col;
    std::cout << std::endl;
    for (int i=0;i<n;i++) std::cout << abc[i].val;
    std::cout << std::endl; 
    return 0;
}
```
