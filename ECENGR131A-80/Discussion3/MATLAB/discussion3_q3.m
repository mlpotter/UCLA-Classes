close all; clear all; clc;

trials = 10000000;
n = 3; p = .6;
for i = 1:trials
flips = rand(1,n)<=p;
Y(i) = sum(flips)-sum(1-flips);
end
%%
clc
Y_set = unique(Y);
for y = Y_set
    fprintf("Experimental Prob of Y=%d : %.5f\n",y,mean(Y==y))
    fprintf("Theoretical Prob of Y=%d : %.5f\n",y,nchoosek(n,(n+y)/2)*p^((n+y)/2) * (1-p)^((n-y)/2))
    fprintf("\n")
end
fprintf("Experimental Mean %.5f\n",mean(Y))
fprintf("Theoretical Mean %.5f\n",2*n*p-n)
%%
fprintf("Experimental Var %.5f\n",var(Y))
fprintf("Theoretical Var %.5f\n",4*n*p*(1-p))