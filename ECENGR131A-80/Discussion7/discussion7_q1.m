close all; clear all; clc;
%%
mu1 = 1; N = 100000;
mu2 = 1/2;
X = exprnd(mu1,1,N);
Y = exprnd(mu2,1,N);
mean(abs(X-Y))
%%