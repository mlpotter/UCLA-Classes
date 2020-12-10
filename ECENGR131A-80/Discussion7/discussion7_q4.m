%%
close all; clc; clear all;
X = rand(1,100000);
Y = rand(1,100000)*2;
Z = X+Y;

figure(1)
hold on
histogram(X,'Normalization','pdf')
histogram(Y,'Normalization','pdf')
histogram(Z,'Normalization','pdf')
%%
z = linspace(0,3,1000)
plot(z,CDF_Z(z),'linewidth',3)
plot(z,PDF_Z(z),'linewidth',3)

%%
function pdf_z = PDF_Z(z)
    pdf_z = ((z >= 0) & (z <= 1)) .* (z)*.5 + ((z > 1) & (z <= 2)) .* (1/2) + ...
         ((z > 2) & (z <= 3)).*(3-z)/2
end

function cdf_z = CDF_Z(z)
    cdf_z = ((z >= 0) & (z <= 1)) .* (z.^2 - z.^2 /2)*.5 + ((z > 1) & (z <= 2)) .* (1/4+(1/2*(z-1))) + ...
         ((z > 2) & (z <= 3)).*(3/4 + .5 * ((z-2).^2 / 2 + ((z-2) - (z-2).^2)))
end