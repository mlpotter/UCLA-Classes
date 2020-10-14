clear all; close all; clc;
number = 30030
count = 0
for i = 1:number
   if mod(number,i)==0 & mod(i,2) == 0
       count = count + 1;
       divisor(count) = i;
   end
end
length(unique(divisor))