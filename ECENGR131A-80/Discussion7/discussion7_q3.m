close all; clc; clear all;
X = rand(1,100000);
Y = rand(1,100000);
Z = X+Y;

figure(1)
hold on
histogram(X,'Normalization','pdf')
histogram(Y,'Normalization','pdf')
histogram(Z,'Normalization','pdf')
%% 
z = linspace(0,2,1000)
plot(z,CDF_Z(z),'linewidth',2)
plot(z,PDF_Z(z),'linewidth',2)
%%
function pdf_z = PDF_Z(z)
    pdf_z = ((z >= 0) & (z <= 1)) .* (z) + ((z > 1) & (z <= 2)) .* (2-z)
end

function cdf_z = CDF_Z(z)
    cdf_z = ((z >= 0) & (z <= 1)) .* (z.^2 - z.^2 /2) + ((z > 1) & (z <= 2)) .* (1/2 - 1/2*(z-1).^2 + (z-1))
end
%%
