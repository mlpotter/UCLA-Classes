close all; clear all; clc;
r_marbs = repmat([0],1,5);
b_marbs = repmat([1],1,5);
marbles = [r_marbs, b_marbs];

trials = 1000000;
for i = 1:trials
   draw = marbles(randperm(length(marbles),2));
   cash = (draw(1)==draw(2))*1.10 - (draw(1)~=draw(2))*1;
   rewards(i) = cash;
end
%%
fprintf("Experimental Mean %.5f\n",mean(rewards))
fprintf("Experimental Var %.5f\n",var(rewards))

mu = 1.10*nchoosek(5,2)/nchoosek(10,2) + nchoosek(5,2)/nchoosek(10,2)*1.10 - 25/nchoosek(10,2);
fprintf("Theoretical Mean %.5f\n",mu)
var_ = nchoosek(5,2)*1.10^2 / nchoosek(10,2) * 2 + 25/nchoosek(10,2) - mu^2;
fprintf("Theoretical Var %.5f\n",var_)