close all; clc; clear all;
x = linspace(-2,2,100);
f = x.^4;
hold on;
plot(x,f);
plot(x,4.*x.^3);
plot(x,12*x.^2);
legend(["f","f^'","f^{''}"])

cosd(45).^2
sind(45).^2