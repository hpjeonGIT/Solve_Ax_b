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
- `cd sparse`
- `mkdir build; cd build`
- `cmake .. -DCMAKE_CXX_COMPILER=mpicxx`
- `make -j 3`
- `make install`
- `./bin/sparse_solver` # Running the solver executable
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
- Code building
  - `mkdir build; cd build; cmake .. ; make -j 4; make install`
  - Link to Google test is necessary. Adjust CMakeLists.txt for the location of libraries
  - Will use C++11 features
  - nvcc and gcc with v> 4.6 will be necessary

## Problem configuration
- Matrix A
  - Sparse symmetric/unsymmetric matrix
  - Positive definiteness not gauranteed
- Vector b
- Wil solve Ax=b, printing x vector as results

## CuSparse sample code
```
```

## HYPRE sample code

```
```
## Note for cuda libraries
- CuBLAS for dense matrix operation such as matmul. No solver
- CuSparse for sparse matrix operation such as matmul. No solver
- CuSolver for solver. Both of dense and sparse matrices
- MPI support for CuSolver is not yet
- For distributed-GPU solver, AmgX is the only available option as of Q1-2020
