close all; clc; clear all; 

Fxy = @(x,y) (1-1/x^2)*(1-1/y^2)
Fxy(1000,1000) - Fxy(4,1000) - Fxy(1000,3) + Fxy(4,3)