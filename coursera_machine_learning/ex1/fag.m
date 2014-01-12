data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
J = computeCost(X, y, theta);

iterations = 1500;
alpha = 0.01;

[theta,J_history] = gradientDescent(X, y, theta, alpha, iterations);
theta
J_history
