close all; clear all; clc;
box = zeros(1,20);
box(1:4) = 1;

trials = 100000;
for i = 1:trials
    picks = randperm(20,3);
    e_df(i) = sum(box(picks));
end
fprintf("Theoretical Solution %.4f\n",mean(e_df))
%%
sum = 0;
for k = 0:3
    sum = sum + nchoosek(4,k)*nchoosek(16,3-k) / nchoosek(20,3) * k;
end
fprintf("Actual Solution %.4f\n",sum)