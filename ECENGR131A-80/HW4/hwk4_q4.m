close all; clear all; clc;
U = linspace(0,1,100)
X = -log(4-4*U)
figure
plot(X,U)
%%
figure
x = linspace(-log(4),10,100000)
plot(x,exp(-x)/4)