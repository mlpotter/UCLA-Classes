close all; clear all; clc;
for n = 1:100
    fprintf("n=%d , <= 0.01, %d\n",n,0.01 >=normcdf(15,n,sqrt(n)))
end