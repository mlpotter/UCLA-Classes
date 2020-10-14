close all; clear all; clc;
trials = 100000;

family_choice = datasample(1:4,trials,'Weights',[.1 .25 .35 .3]);
for i = 1:length(family_choice)
    child_selection(i) = randperm(family_choice(i),1);
end

experimental = mean((child_selection==1) &(family_choice==1)) / mean(child_selection == family_choice);
theoretical = .1/(.1+.5*.25 + 1/3*.35 + 1/4*.3);

fprintf("Experimental %.4f\n",experimental)
fprintf("Theoretical %.4f\n",theoretical)
%%
theoretical = ((1/4)*.3)/(.1 + .5*.25 + 1/3*.35 + 1/4*.3);
experimental = mean((child_selection==4) & (family_choice==4)) / mean(child_selection == family_choice);
fprintf("Experimental %.4f\n",experimental)
fprintf("Theoretical %.4f\n",theoretical)