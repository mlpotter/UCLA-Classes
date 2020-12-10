close all; clc; clear all;
X = rand(1,100000);
Y = rand(1,100000);
Z = X+Y;

figure(1)
hold on
histogram(X,'Normalization','pdf')
histogram(Y,'Normalization','pdf')
histogram(Z,'Normalization','pdf')
%%
close all; clc; clear all;
X = rand(1,100000);
Y = rand(1,100000)*2;
Z = X+Y;

figure(1)
hold on
histogram(X,'Normalization','pdf')
histogram(Y,'Normalization','pdf')
histogram(Z,'Normalization','pdf')