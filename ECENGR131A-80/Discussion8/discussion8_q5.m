close all; clear all; clc;
%%
mu_ = 2.4;
std_ = 2;

n = 100;

mu_sn = 2.4*n;
std_sn = sqrt(std_^2 * n);

1-qfunc((250-mu_sn)/std_sn)
%%
Sn = sum(normrnd(mu_,std_,1000000,n),2);
mean(Sn <= 250)

