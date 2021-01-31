clear all; clc; close all;
mu = [0 0]; Sigma = [3 .7; .7 2];
[X1,X2] = meshgrid(linspace(-3,3,50)', linspace(-3,3,50)');
X = [X1(:) X2(:)];
pX = mvnpdf(X, mu, Sigma);
Xr = mvnrnd(mu,Sigma,5000);

figure
hold on
contour(X1,X2,reshape(pX,50,50))


[U,S,V] = svd(Xr,0);
% V = V'

projection = U*S;

quiver(0,0,V(1,1),V(2,1))
quiver(0,0,V(1,2),V(2,2))

%%

A = [1 2 3; 2 3 4]
[U,D,V] = svd(A)

A*V(:,1)
U(:,1) * D(1,1)

A'*U(:,1)
V(:,1)*D(1,1)
%%



