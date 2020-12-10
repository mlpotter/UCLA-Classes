close all; clear all; clc;

mu_ = 50; std_ = 5;
(1-qfunc((60-mu_)/std_)) - (1-qfunc((40-mu_)/std_))

(1-qfunc((53-mu_)/std_)) - (1-qfunc((50-mu_)/std_))
%% simulation
X = sum(rand(100000,100) <= 0.5,2);
mean((X <= 60) & (X > 40))
mean((X <=53) & (X > 50))




