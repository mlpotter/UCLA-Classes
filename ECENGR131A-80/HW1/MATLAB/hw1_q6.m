close all; clear all; clc;
N = 5000000;
trials = randi([1,6],N,6);

s1 = sum(trials == 1,2)==2;
s2 = sum(trials == 2,2)==2;
s3 = sum(trials == 3,2)==2;
s4= sum(trials == 4,2)==2;
s5 = sum(trials == 5,2)==2;
s6 = sum(trials == 6,2)==2;

subset = (sum([s1 s2 s3 s4 s5 s6],2)==3);

fprintf("Experimental Solution %.4f\n",sum(subset)/N)
fprintf("Theoretical Solution %.4f\n",(nchoosek(6,3)*factorial(6)/(2^3))/(6^6))
%%
all_cases = unique(trials,'rows');
%%
s1 = sum(all_cases == 1,2) == 2;
s2 = sum(all_cases == 2,2) == 2;
s3 = sum(all_cases == 3,2) == 2;
s4 = sum(all_cases == 4,2) == 2;
s5 = sum(all_cases == 5,2) == 2;
s6 = sum(all_cases == 6,2) == 2;
mean(sum([s1,s2,s3,s4,s5,s6],2)==3)
%%