%% b
close all; clear all; clc;
trials = 1000000;
count = 0;

flips = rand(trials,4) < .5;
fprintf("B. Experimental %.4f\n",mean(sum(flips(:,1:3),2)==1))
%% a
close all;
count = 0;
flips = rand(trials,4) > .5;
for i = 1:length(flips)
    experiment = flips(i,:);
    if sum(experiment) == 4
        count = count + 1;
        continue
    end
    if (experiment(1)&experiment(2)&experiment(3) == 1) | (experiment(2)&experiment(3)&experiment(4) == 1)
        count = count + 1;
        continue
    end
    if (experiment(1)&experiment(2) == 1) | (experiment(2)&experiment(3) == 1)  | (experiment(3)&experiment(4) == 1)
       count = count + 1;
       continue
    end
end
fprintf("A. Experimental %.4f\n",count/trials)

