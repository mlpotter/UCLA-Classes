X = rand(1,100000)*2;
Y = rand(1,100000)*4;
Z = X+Y;
histogram(Z,'Normalization','pdf')

hold on 
z = linspace(0,6,10000)
pdf_z = PDF_Z(z)
plot(z,pdf_z)

function pdf_z = PDF_Z(z)
    r1 = (z <= 2) & (z >= 0)
    r2 = (z < 4) & (z > 2)
    r3 = (z < 6) & (z >= 4)
    pdf_z = r1.*z/8 + r2.*1/4 + r3.*(6-z)/8
end