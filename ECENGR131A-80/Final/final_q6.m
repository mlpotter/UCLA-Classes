close all; clear all; clc;
mux = 0;
muy = 0;
rho = 0;
sigmax = 1/2
sigmay = 1

syms x y
p = -1/2 * (((x-mux)/sigmax)^2 - 2*rho*(x-mux)/sigmax * (y-muy)/sigmay + ((y-muy)/sigmay)^2) * (1/(1-rho^2));
expand(p)
%%