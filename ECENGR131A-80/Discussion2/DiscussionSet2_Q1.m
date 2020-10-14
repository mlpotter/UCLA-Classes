close all; clear all; clc;
trials = 100000
MC = rand(trials,1000)<.5;
count = 0
for i = 1:size(MC,1)
   first = find(MC(i,:));
   first = first(1);
   if mod((first+1),2)==0
       count = count + 1;
   end
end

fprintf("Experimental %.5f\n",count/trials)