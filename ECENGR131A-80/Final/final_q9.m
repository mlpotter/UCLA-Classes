close all; clear all; clc;
p_exact = 0
for i = 230:370
    p_exact = p_exact + nchoosek(1000,i)*.2^i * .8^(1000-i);
end

p_approximate = (1-qfunc((370-200)/sqrt(160))) - (1-qfunc((230-200)/sqrt(160)));
%% simulation

bit_sent = ((rand(100000,1000) < .5)-.5)*2;
bit_flip = ((rand(100000,1000) < .2)-.5) * -2;
bit_received = bit_sent .* bit_flip;
total_errors = sum(bit_sent ~= bit_received,2);
p_experimental = mean((total_errors <= 370) & (total_errors >= 230));

array2table([p_exact p_approximate p_experimental],'VariableNames',["Exact","Approx","Exp"])