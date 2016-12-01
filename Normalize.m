function [ X_norm ] = Normalize( X )
  %
  % Mon Feb 15 06:37:12 2016 written by Chelsea Weaver
  %
  % Normalize(X) normalizes the columns of the matrix X to have l2-norm equal
  % to 1.

  [m,~] = size(X);

  Norms = sqrt(sum(X.^2,1)); % row vector of column norms
  X_norm = X./repmat(Norms,m,1);

  % Replace any NaN's with 0's:
  a = find(Norms==0);
  X_norm(:,a) = zeros(m,length(a));

end

