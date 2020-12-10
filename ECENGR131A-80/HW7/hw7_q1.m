close all; clear all; clc;
x = linspace(0,1,10000);
y = linspace(1,5,10000);
fxy = @(x,y) x/5 + y/20;
fx = @(x) (4*x + 3)/5;
fy = @(y) (2+y)/20;
gxy = fxy(x,y);
zxy = fx(x) .* fy(y);
%%
plot3(x,y,gxy);
figure();
plot(fx(x),fy(y));


