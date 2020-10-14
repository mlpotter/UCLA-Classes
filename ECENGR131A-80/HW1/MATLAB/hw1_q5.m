%% distinguishable balls and distinguishable bins
close all; clear all; clc;
N = 8;
trials = 100000;
for i = 1:trials
    b = [];
    balls = randi([1,N],1,N);
    for j = 1:N
        b(j) = sum(balls==j);
    end
    
    ball_bin(i,:) = b;
end
%%
numerator = sum(sum(ball_bin == 0,2)==1);
fprintf("A. Experimental Solution %.4f \n",numerator/trials)
%% our solution
fprintf("A. Theoretical Solution %.4f \n",(N*(N-1)*nchoosek(N,2)*factorial(N-2))/(N^N))
fprintf("A. Theoretical Solution %.4f \n",(nchoosek(N,2)*factorial(N))/(N^N))
%% undistinguishable balls and distinguishable bins
close all; clear all; 
N = 8;
trials = 100000;

for i = 1:trials
    b = [];
    positions = ones(1,N+N-1);
    dividers = sort(randperm(N+N-1,N-1));
    positions(dividers) = 0;
    b(1) = sum(positions(1:dividers(1)));
    for j = 1:(length(dividers)-1)
        b(j+1) = sum(positions(dividers(j):dividers(j+1)));
        
    end
    b(length(dividers)+1) = sum(positions(dividers(end):end));
    ball_bin(i,:) = b;
    
end
%%
numerator = sum(sum(ball_bin == 0,2)==1);
fprintf("B. Experimental Solution %.4f \n",numerator/trials)
%% our solution
fprintf("B. Theoretical Solution %.4f \n",(N*(N-1))/(nchoosek(N+N-1,N-1)))
%% - %% - %% - %% - %%
%% - %% - %% - %% - %%
%% - %% - %% - %% - %%