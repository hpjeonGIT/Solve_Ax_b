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
A(1,10) = 1; A(2,10)=2
%A =
%
%   10    8    0    0    0    0    0    0    0    1
%    7   10    8    0    0    0    0    0    0    2
%    0    7   10    8    0    0    0    0    0    0
%    0    0    7   10    8    0    0    0    0    0
%    0    0    0    7   10    8    0    0    0    0
%    0    0    0    0    7   10    8    0    0    0
%    0    0    0    0    0    7   10    8    0    0
%    0    0    0    0    0    0    7   10    8    0
%    0    0    0    0    0    0    0    7   10    8
%    0    0    0    0    0    0    0    0    7   10
x = inv(A)*b
% -1.922771   2.324830  -1.380881   0.066875   1.624677  -1.464362   1.158860   
%  0.707741  -0.898680   1.629076
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