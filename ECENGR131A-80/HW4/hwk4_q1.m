close all; clear all; clc;
beta = [.5,1,2]; lambda = 1;
x = linspace(0,10,1000);
figure
hold on
for b = beta
    plot(x,1-exp(-(x./lambda).^b),'linewidth',3)
end
legend(["\lambda=1 \beta=.5","\lambda=1 \beta=1","\lambda=1 \beta=2"],'fontsize',15)
xlabel("x",'fontsize',20)
ylabel("1-e^{-(x/\lambda)^\beta}",'fontsize',20)
title("Weibull CDF",'fontsize',20)
%%
close all; clear all; clc
figure
hold on
beta = 2; lambda = 1;
x = linspace(0,10,1000);
weibull = exp(-(x./lambda).^beta);
plot(log(x),log(weibull),'linewidth',3)
xlabel("ln x",'fontsize',20)
ylabel("ln e^{-(x/\lambda)^\beta}",'fontsize',20)
title("ln e^{-(x/\lambda)^\beta} vs ln x",'fontsize',15)
