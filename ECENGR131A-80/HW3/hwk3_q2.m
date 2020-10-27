close all; clear all; clc;
p = 1/4;
for x = 1:20
    fprintf("%d %.10f %d\n",x,(1-p)^(x),0.01 >= (1-p)^(x))
end