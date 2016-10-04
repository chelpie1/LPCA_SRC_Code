function [ A_norm ] = Normalize( A )
%
% Mon Feb 15 06:37:12 2016 written by Chelsea Weaver
%
% Normalize(A) normalizes the columns of the matrix A to have l2-norm equal
% to 1.

[m,~] = size(A);

Norms = sqrt(sum(A.^2,1)); % row vector of column norms
A_norm = A./repmat(Norms,m,1);

% Replace any NaN's with 0's:
a = find(Norms==0);
A_norm(:,a) = zeros(m,length(a));

end

