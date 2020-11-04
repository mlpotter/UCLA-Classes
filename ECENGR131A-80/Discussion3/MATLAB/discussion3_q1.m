close all; clear all; clc;
o_balls = repmat([0],1,8);
b_balls = repmat([1],1,4);
w_balls = repmat([2],1,2);
balls = [o_balls, b_balls, w_balls];

trials = 100000
for i = 1:trials
   draw = balls(randperm(length(balls),2));
   cash = sum(draw==0)*0 + sum(draw==1)*2 - sum(draw==2)*1;
   rewards(i) = cash;
end
%%
mean(rewards==1)