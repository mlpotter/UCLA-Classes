%%
close all; clear all; clc;
A = 1/(sqrt(2))*[1 1;1 -1]
A*A'
eig(A)

det(A)

[V,D] = eig(A)
%%
lam = 1;

v1 = [1+sqrt(2);1];
v1 = v1/norm(v1)

A*v1 - lam*v1
%%
lam = -1;
v2 = [1-sqrt(2) ; 1];
v2 = v2/norm(v2)
A*v2 - lam*v2