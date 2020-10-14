close all; clear all; clc;
n1 = 3;
n2 = 2;
m = 5;

trials = 100000;
for i = 1:trials
    b = [];
    positions = ones(1,n1+m-1);
    dividers = sort(randperm(n1+m-1,m-1));
    positions(dividers) = 0;
    b(1) = sum(positions(1:dividers(1)));
    for j = 1:(length(dividers)-1)
        b(j+1) = sum(positions(dividers(j):dividers(j+1)));
    end
    b(length(dividers)+1) = sum(positions(dividers(end):end));

    w = [];
    positions = ones(1,n2+m-1);
    dividers = sort(randperm(n2+m-1,m-1));
    positions(dividers) = 0;
    w(1) = sum(positions(1:dividers(1)));
    for j = 1:(length(dividers)-1)
        w(j+1) = sum(positions(dividers(j):dividers(j+1)));
    end
    w(length(dividers)+1) = sum(positions(dividers(end):end));

    ball_bin(i,:) = w+b;
    
end
%%
length(unique(ball_bin,'rows'));
%%
fprintf("Experimental Solution %.4f\n",mean(sum(ball_bin > 0,2)==m))
%% 
num = 0;
for i = 1:(m-1)
    num = num + (-1)^(i+1)*nchoosek(m,i)*nchoosek(n1+(m-i)-1,(m-i)-1)*nchoosek(n2+(m-i)-1,(m-i)-1);
end
% num =  m*nchoosek(n1+(m-1)-1,(m-1)-1)*nchoosek(n2+(m-1)-1,(m-1)-1) -  nchoosek(m,2)*nchoosek(n1+(m-2)-1,(m-2)-1)*nchoosek(n2+(m-2)-1,(m-2)-1) + nchoosek(m,3)*nchoosek(n1+(m-3)-1,(m-3)-1)*nchoosek(n2+(m-3)-1,(m-3)-1);
den = nchoosek(n1+m-1,m-1)*nchoosek(n2+m-1,m-1);
fprintf("Theoretical Solution %.4f\n",1-num/den)
%%
num = 0
for i = 1:m-1
    num = num + (-1)^(i) * nchoosek(m,i) * nchoosek(n1+m-i-1,n1)*nchoosek(n2+m-i-1,n2)
end
1 + num/den