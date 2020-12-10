close all; clear all; clc;
mu = 1; N = 10000;
X = exprnd(mu,1,N);
Y = exprnd(mu,1,N);
Z = X./(X+Y);

figure(1)
hold on
histogram(X,'Normalization','pdf') 
histogram(Y,'Normalization','pdf') 
histogram(Z,'Normalization','pdf') 