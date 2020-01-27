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

## Matrix market format
```
1 1 x
2 1 x
3 1 x
2 2 x
3 2 x
...
```
First column is the row, and the second column is the column index. CSR splits t
he matrix elements along row index, and the appropriate form would be:
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
