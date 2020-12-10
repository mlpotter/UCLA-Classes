close all; clear all; clc;
mux = 1;
muy = 0;
rho = 3/4
sigmax = sqrt(1/(1-rho^2))
sigmay = sqrt(1/(4*(1-rho^2)))

syms x y
p = (((x-mux)/sigmax)^2 - 2*rho*(x-mux)/sigmax * (y-muy)/sigmay + ((y-muy)/sigmay)^2) * (1/(1-rho^2));
expand(p)
%%