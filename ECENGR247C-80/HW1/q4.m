close all; clear all; clc;
N = 1000;
W = randn(2,2)
x = rand(N,2)';
alpha = 0.05;
y = W*x + mvnrnd([0,0],[1,0;0 1]*alpha,N)';

y*x' * inv(x*x')

Y = y'; 
X = x';

(Y'*X)*inv(X'*X)

x1 = [1;1]; y1 = [2;1];
x2 = [1;2]; y2 = [3;4];

(x1*y1' + x2*y1')'
(x1*y1')' + (x2*y1')'