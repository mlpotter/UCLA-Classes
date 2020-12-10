close all; clear all; clc;
n = 1:100;
p = .4;
alpha = 1;
upper_bound = 1./n * p*(1-p)
stem(n,upper_bound)
title("Chebyshev Upper Bound vs n",'FontWeight','bold')
xlabel("n",'FontWeight','bold')
ylabel("Upper Bound",'FontWeight','bold')
%%
n = 50;
Y = rand(100000,50);
sum(abs(mean(Y < .4,2)-.4) == 1)