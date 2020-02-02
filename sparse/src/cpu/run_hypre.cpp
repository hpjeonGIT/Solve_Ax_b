#include <cmath>
#include <iostream>
#include <vector>
#include <mpi.h>
#include "run_hypre.h"
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

void HYPRE_solver::run_hypre(mtrx_csr &spdata, rhs &b_v, int const &myid) {
    int ilower, iupper, local_size;
    HYPRE_IJMatrix A;
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVector x;
    HYPRE_ParVector par_x;
    HYPRE_Solver solver, precond;
    /* Default problem parameters */
    ilower = spdata.ilower_;
    iupper = spdata.iupper_;
    local_size = spdata.local_size_;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
    // Note that this is for a symmetric matrix, ilower/iupper of row and ilower/iupper of column are same
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);
    HYPRE_IJMatrixSetValues(A, local_size, &spdata.nnz_v_[0], &spdata.rows_[0],
            &spdata.colidx_[0], &spdata.values_[0]);
    HYPRE_IJMatrixAssemble(A);
    /*
    for (int i=0; i< spdata.rows_.size(); i++ ) {
    std::cout << " rows_ " << spdata.rows_[i] ;
    }
    for (int i=0; i< spdata.colidx_.size(); i++ ) {
    std::cout << " coldix_ " << spdata.colidx_[i] ;
    }
    for (int i=0; i< spdata.nnz_v_.size(); i++ ) {
    std::cout << " nnz_v_ " << spdata.nnz_v_[i] ;
    }
    for (int i=0; i< b_v.rows_.size(); i++ ) {
        std::cout << " b_v.rows_ " << b_v.rows_[i] << " " << b_v.values_[i] << std::endl;
        }
*/
/*  10  8 0 0
 *   8 10 8 0
 *   0 8 10 8
 *   0 0 8 10
 */

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
    x_values_.resize(b_v.rows_.size(), 0.0);
    HYPRE_IJVectorSetValues(b, b_v.rows_.size(), &b_v.rows_[0], &b_v.values_[0]);
    HYPRE_IJVectorSetValues(x, b_v.rows_.size(), &b_v.rows_[0], &x_values_[0]);
    //b_v.values_.clear();
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
    HYPRE_FlexGMRESSetMaxIter(solver, 10); /* max iterations */
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
    HYPRE_IJVectorGetValues(x, b_v.rows_.size(), &b_v.rows_[0], &x_values_[0]);

    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(b);
    HYPRE_IJVectorDestroy(x);
    std::cout << "myid = " << myid << " x= ";
    for (int i=0; i< local_size;i++) {
        std::cout << x_values_[i] << " ";
    }
    std::cout << std::endl;
}


void HYPRE_solver::get_result(std::vector<double> &x){
    x.resize(x_values_.size());
    x = x_values_;
}
