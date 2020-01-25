## A sample for a symmetric sparse matrix

A =  
   10    8    0    0    0    0    0    0    0    0     
    8   10    8    0    0    0    0    0    0    0      
    0    8   10    8    0    0    0    0    0    0      
    0    0    8   10    8    0    0    0    0    0      
    0    0    0    8   10    8    0    0    0    0      
    0    0    0    0    8   10    8    0    0    0      
    0    0    0    0    0    8   10    8    0    0  
    0    0    0    0    0    0    8   10    8    0      
    0    0    0    0    0    0    0    8   10    8    
    0    0    0    0    0    0    0    0    8   10 

 b =  
    1    2    3    4    5    6    7    8    9   10  

For MPI rank=0  
A0=  
   10    8    0    0    0    0    0    0    0    0      
    8   10    8    0    0    0    0    0    0    0    
    0    8   10    8    0    0    0    0    0    0  
    0    0    8   10    8    0    0    0    0    0  

For MPI rank=1  
A1=  
    0    0    0    8   10    8    0    0    0    0  
    0    0    0    0    8   10    8    0    0    0  
    0    0    0    0    0    8   10    8    0    0  

For MPI rank=2  
A2=   
    0    0    0    0    0    0    8   10    8    0  
    0    0    0    0    0    0    0    8   10    8  
    0    0    0    0    0    0    0    0    8   10  

- A sample octave/matlab code is shown:
```
N = 10
Nnear = N*0.7
A = zeros(N,N)
b = zeros(N,1)
% well-defined-symmetric
for i=1:N
  for j=1:N
    tmp = N - 2*abs(i-j);
    if (tmp > Nnear)
      A(i,j) = tmp;
    endif
   endfor  
   b(i) = i;
endfor
x = inv(A)*b
```

x =  
0.834992  -0.918740   0.563433   0.589449  -0.800244   1.035856   0.255424  -0.480136 1.344746  -0.075797  

- Running HYPRE sample code:
    - `export HYPRE_ROOT=/home/hpjeon/sw_local/hypre/2.18.2` Or configure as your HYPRE install PREFIX
    - `mpicxx -o a.exe -std=c++11 hypre_ex.cpp -I$HYPRE_ROOT/include -L$HYPRE_ROOT/lib -lHYPRE`
    - `mpirun -np 3 ./a.exe`
- The sample HYPRE code has hard-coded 3-partitions for 3 MPI ranks. It yields same answer of Octave
```
myid = 0 x= 0.834992 -0.91874 0.563433 0.589449
myid = 1 x= -0.800244 1.03586 0.255424 
myid = 2 x= -0.480136 1.34475 -0.0757966 
```
- Running Amgx sample code
    - `export AMGX_ROOT=/home/hpjeon/sw_local/amgx`
    - `mpicxx -std=c++11 amgx_ex.cpp -I/usr/local/cuda/include -I${AMGX_ROOT}/include -L/usr/local/cuda/lib64 -lcudart -L${AMGX_ROOT}/lib -lamgxsh`
    - `mpirun -np 3 ./a.out`
- The sample Amgx code had the same 3-partitions of HYPRE example. Answers match each other 
```
my id = 0 x= 0.834992 -0.91874 0.563433 0.589449
my id = 1 x= -0.800244 1.03586 0.255424
my id = 2 x= -0.480136 1.34475 -0.0757966
```


    