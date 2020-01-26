N = 10
Nnear = N*0.7
A = zeros(N,N)
b = zeros(N,1)
% well-defined-symmetric
for i=1:N
  for j=1:N
    tmp = N - 2*abs(i-j);
    if (tmp > Nnear)
      if ( i > j)
        tmp = tmp - 1;
        endif
      A(i,j) = tmp;      
    endif
   endfor  
   b(i) = i;
endfor
x = inv(A)*b
% -3.25058   4.18823  -2.14103  -0.61342   3.14017  -2.76347   1.45669   1.47217
%  -2.11482   2.48038
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