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
        std::vector<int> cols;
        int nnz;
        if (myid == 0){
         /*    10    8    0    0    0    0    0    0    0    1
                7   10    8    0    0    0    0    0    0    2
                0    7   10    8    0    0    0    0    0    0
                0    0    7   10    8    0    0    0    0    0 */
            values = {10., 8., 1.}; cols = {0,1,9}; nnz = 3; n=0;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {7., 10., 8., 2.}; cols = {0,1,2,9}; nnz = 4; n=1;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {7., 10., 8.}; cols = {1,2,3}; nnz = 3; n=2;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {7., 10., 8.}; cols = {2,3,4}; nnz = 3; n=3;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
        } else if (myid == 1) {
        /*  0    0    0    7   10    8    0    0    0    0
            0    0    0    0    7   10    8    0    0    0
            0    0    0    0    0    7   10    8    0    0 */
            values = {7., 10., 8.}; cols = {3,4,5}; nnz = 3; n=4;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {7., 10., 8.}; cols = {4,5,6}; nnz = 3; n=5;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {7., 10., 8.}; cols = {5,6,7}; nnz = 3; n=6;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
        } else {// myid == 2
       /*   0    0    0    0    0    0    7   10    8    0
            0    0    0    0    0    0    0    7   10    8
            0    0    0    0    0    0    0    0    7   10 */
            values = {7., 10., 8.}; cols = {6,7,8}; nnz = 3; n=7;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {7., 10., 8.}; cols = {7,8,9}; nnz = 3; n=8;
            HYPRE_IJMatrixSetValues(A, 1, &nnz, &n, &cols[0], &values[0]);
            values = {7., 10.};      cols = {8,9}; nnz = 2; n=9;
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
