close all; clear all; clc;
p = 0.2; N = 5;

bits_encoded = rand(5,1) < .5; % <.5 is 1, >= .5 is 0 transmitted
bits_sent = rand(5,N);
zero_bits = (bits_sent < p) .* (1-bits_encoded);
one_bits = (1-(bits_sent < p)) .* bits_encoded;
%%
bits_encoded
bits_sent
zero_bits
one_bits
%% 
bits_sent = zero_bits + one_bits;