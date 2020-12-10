close all; clear all; clc;
global rho
global sigma

rho = 1.1;
sigma = .5;

n = 12;

var_sn = 0;
for i = 1:n
    for j = 1:n
        if i == j
            continue
        else
            var_sn = var_sn + COV(i,j);
        end
    end
end
          
var_sn = var_sn + n*sigma^2

n*sigma^2 + 2*(sigma^2*rho)/(1-rho) * ((n-1) - rho*(1-rho^(n-1))/(1-rho))

function cov = COV(i,j)
global rho
global sigma
cov = rho^(abs(i-j)) * sigma^2;
end