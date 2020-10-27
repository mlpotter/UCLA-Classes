close all; clear all; clc;

trials = 1000000;

for i = 1:trials
roll = randi([1,3],3);
X1 = sum(roll(1:2),2);
X2 = roll(3);
pdf(i) = X1-X2;
end
%%
for i = -1:5
   fprintf("%d %.5f \n",i,mean(pdf==i))
end
%%
for i = 0:5
   abolute = abs(pdf);
   fprintf("%d %.5f \n",i,mean(abolute>=i))
end
%% pdf
hist(pdf)
%% mean and variance
mean(pdf)
var(pdf)
%%
clear all; close all; clc;
count = 1
for i = 1:3
    for j = 1:3
        for k = 1:3
            fprintf("%d %d %d\n",i,j,k)
            roll(count,:) = [i,j,k]
            count = count + 1
        end
    end
end
%%
%%
Y = sum(roll(:,1:2),2) - roll(:,3)
for i = unique(Y)'
   fprintf("P(Y==%d)=%.4f\n",i,mean(Y==i)) 
end
fprintf("\n")
for i = unique(Y)'
   fprintf("P(|Y|>=%d)=%.4f\n",abs(i),mean(abs(Y)>=(i)))
end
%%
for i = -1:5
    fprintf("%d %d\n",i,sum((sum(roll(:,1:2),2) - roll(:,3))==i))
end
EX = -1/27 + 0*3/27 + 6/27 + 14/27 + 18/27 + 12/27 + 5/27
VARX = 1/27*(-1-2)^2 + 3/27*(0-2)^2 + 6/27*(1-2)^2 + 7/27*(2-2)^2 + 6/27*(3-2)^2 + 3/27*(4-2)^2 + 1/27*(5-2)^2