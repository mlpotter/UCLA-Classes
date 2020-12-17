%% a
close all; clc; clear all;
samples = [1,3,10,30];

i = 1;
for n = samples
    xs = randi([1,4],10000,n);
    Z = sum(xs,2);
    fprintf("Z @ n =%3d = %4.5f ; =%4.5f\n",n,mean(Z),var(Z))
    z(i,:) = Z;
    i = i + 1;
end
%%
figure(1)
hold on
for i = 1:length(samples)
   [C,ia,ic] = unique(z(i,:));
    a_counts = accumarray(ic,1);
    value_counts = [C', a_counts];
    stem(C',a_counts/sum(a_counts));
end
legend("n="+string(samples),'FontSize',20)
fprintf("\n\n")
ylabel("P[Zn = z]",'FontSize',20)
xlabel("Zn",'FontSize',20)
%% b 
clearvars -except samples

i = 1;
for n = samples
    mu = (4+1)/2;
    var = ((4-1+1)^2 - 1) / 12;
    z_mu(i) = (mu*n);
    z_var(i) = (var*n);
    fprintf("Z_mu @ n = %3d = %4.5f\n",n,z_mu(i))
    fprintf("Z_var @ n = %3d = %4.5f\n\n",n,z_var(i))
    i = i + 1;
end
%% c
figure(1)
hold on
for i = 1:length(samples)
x = linspace(z_mu(i) - sqrt(z_var(i))*3,z_mu(i) + sqrt(z_var(i))*3,1000);
y = normpdf(x,z_mu(i),sqrt(z_var(i)));
plot(x,y,'linewidth',3)
end
legend('CLT n=1','CLT n=3','CLT n=10','CLT n=30','PDF n=1','PDF n=3','PDF n=10','PDF n=30')