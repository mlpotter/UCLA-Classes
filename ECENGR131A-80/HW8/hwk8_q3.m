close all; clear all; clc;
n = 11;
z_mu = n*47000+74000 - (n*50000);
z_var = n * 10000^2;

1-qfunc((20000-z_mu)/(sqrt(z_var)))
%% b
X = (20000 - qfuncinv(1-0.005) * sqrt(z_var) - 74000 + 11*50000)/11
%% simulation
gas_left = 74000 + sum(47000 - normrnd(50000,10000,1000000,11),2);
mean(gas_left < 20000)

gas_left = 74000 + sum(X - normrnd(50000,10000,1000000,11),2);
mean(gas_left < 20000)
