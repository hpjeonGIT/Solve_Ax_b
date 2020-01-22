# Code Project
- Sparse Matrix Solver

## Objectives
- Solve Ax = b
- Using HYPRE for CPU
- Using CuSparse for GPU
- TBD: Compare performance - wall time, memory foot-print on each
- Unit-tests for each src folder

## Steps to use
- `cd sparse`
- `mkdir build; cd build`
- `cmake ..`
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
