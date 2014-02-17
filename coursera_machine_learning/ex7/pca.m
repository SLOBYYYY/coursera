function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

% Covariance matrix
Sigma = X' * X / m;

% Compute 'eigenvectors' of Sigma by using Singular Value Decomposition
[U, S, ~] = svd(Sigma);

% U is a n * n matrix that contains all the needed u vectors 
% (u is a column vector). Usually we take the first K 
% (number of principal component) columns that we want the reduce 
% our training example to

% We should choose K depending on how big the retained variance is after
% we applied PCA. We can calculate the retained variance with the following:
% Returned matrix S from svd looks like this:
% [S11,0,0;0,S22,0;0,0,S33]
% We should ensure that the sum of Sii where i=1 to K
% divided by Sii where i=1 to n is bigger then 0.99 (if we aim for 99% retained variance)

% =========================================================================

end
