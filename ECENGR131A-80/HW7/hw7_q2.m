close all; clear all; clc;
%% part a
sigmax = 2.12;
sigman = 5;
X = normrnd(0,sigmax,1,1000000) ;
N = normrnd(0,sigman,1,1000000);
Y = X+N;

pcorrcoeff = sigmax/(sqrt(sigmax^2 + sigman^2))'
corrcoef(X,Y)
%% part b
a = linspace(-5,5,1000);
figure(1)
plot(a,mean((X'-a.*Y').^2));
a_true = sigmax^2 / (sigmax^2 + sigman^2)
%% part c
mse = (sigmax^2 * sigman^2)/(sigmax^2 + sigman^2)'
hold on
plot(a_true,mse,'r*')
ylabel("MSE")
xlabel("a")