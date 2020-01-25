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
% 0.834992  -0.918740   0.563433   0.589449  -0.800244   1.035856   0.255424  -0.480136
% 1.344746  -0.075797
%
% random - nonsymmetric
for i=1:N
  for j=1:N
    tmp = N - 2*abs(i-j);
    if (tmp > Nnear)
      A(i,j) = tmp + rand(1) - 0.5;
    endif
   endfor  
endfor
b = rand(N,1)*10;
x = inv(A)*b;
%   0.044548   0.168754   0.412500   0.536485  -0.713103   1.116942  -0.024728   0.038530
%   0.340994  -0.184016