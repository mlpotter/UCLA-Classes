%% 1a
close all; clear all; clc;
tosses = [10,50,100,500,1000];
for t = tosses
    p_odd = mean(mod(randi(4,1,t),2));
    fprintf("t =%5d , p(odd number)=%.4f\n",t,p_odd)
end
%% 1d
close all; clear all; clc;
tosses = [10,50,100,500,1000];
s = RandStream('mlfg6331_64');
for t = tosses
    p_odd = mean(mod(datasample(s,1:4,t,'Weights',[1/6,2/6,1/6,2/6]),2));
    fprintf("t =%5d , p(odd number)=%.4f\n",t,p_odd)
end