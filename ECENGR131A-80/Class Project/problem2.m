close all; clear all; clc;
%%
p = 0.01:0.01:.1;
N = 1:2:7;

for i = 1:length(N)
    for j = 1:length(p)
        [bits_sent,bits_encoded] = transmitted(p(j),N(i));
        p_error(i,j) = mean(decoder(bits_sent,N(i)) ~= bits_encoded);
        theoretical_error(i,j) = p_error_theor(p(j),N(i));
    end
end
subplot(1,2,1)
hold on
cmap = colormap(parula(length(p)));
for i = 1:length(p)
    plot(N,p_error(:,i),'Color',cmap(i,:),'LineWidth',2)
end
legend("p="+string(p))
title("N-repetition BSC Error Experimental Analysis")
xlabel("N",'FontWeight','bold')
ylabel("P(error)",'FontWeight','bold')

figure(1)
subplot(1,2,2)
hold on
cmap = colormap(parula(length(p)));
for i = 1:length(p)
    plot(N,p_error(:,i),'Color',cmap(i,:),'LineWidth',2)
end
set(gca,'yscale','log')
legend("p="+string(p))
title("N-repetition BSC Error Experimental Analysis")
xlabel("N",'FontWeight','bold')
ylabel("P(error) log axis",'FontWeight','bold')

array2table(p_error,'RowNames',"N="+string(N),'VariableNames',"P="+string(p))

%%
figure(2)
subplot(1,2,2)
hold on
for i = 1:length(p)
    plot(N,theoretical_error(:,i),'Color',cmap(i,:),'LineStyle','--','LineWidth',2)
end
legend("p="+string(p))
set(gca,'yscale','log')
legend("p="+string(p))
title("N-repetition BSC Error Theoretical Analysis")
xlabel("N",'FontWeight','bold')
ylabel("P(error) log axis",'FontWeight','bold')

subplot(1,2,1)
hold on
for i = 1:length(p)
    plot(N,theoretical_error(:,i),'Color',cmap(i,:),'LineStyle','--','LineWidth',2)
end
legend("p="+string(p))
legend("p="+string(p))
title("N-repetition BSC Error Theoretical Analysis")
xlabel("N",'FontWeight','bold')
ylabel("P(error)",'FontWeight','bold')

array2table(theoretical_error,'RowNames',"N="+string(N),'VariableNames',"P="+string(p))


%%
function [bits_sent,bits_encoded] = transmitted(p,N)
    bits_encoded = rand(20000,1) < .5; % <.5 is 1, >= .5 is 0 transmitted
    bits_sent = rand(20000,N);
    bits_sent = (bits_sent < p) .* (1-bits_encoded) + (1-(bits_sent < p)) .* bits_encoded;
end

function bits_received = decoder(transmission,N)
    bits_received = sum(transmission,2)>N/2;
end

function theoretical_error = p_error_theor(p,N)
    n = ceil(N/2);
    theoretical_error = 0;
    for i = n:N
        theoretical_error = theoretical_error + nchoosek(N,i)*p^i*(1-p)^(N-i);
    end
end