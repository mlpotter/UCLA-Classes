close all; clear all; clc;
N = 1000;

mu1 = [1,1]; sigma1 = [1 .2; .2 1];
mu2 = [5,4]; sigma2 = [2 .5; .5 1];

x_given_c1 = mvnrnd(mu1,sigma1,N);
y1 = ones(N,1);

x_given_c2 = mvnrnd(mu2,sigma2,N);
y2 = ones(N,1) * -1;

X = [x_given_c1  ; x_given_c2];
y = [y1  ; y2];

figure 
hold on
plot(x_given_c1(:,1),x_given_c1(:,2),'r*')
plot(x_given_c2(:,1),x_given_c2(:,2),'g*')

[X1,X2] = meshgrid(-2:.5:12,0:.5:8);
X_mesh = [X1(:) X2(:)];

w = randn(2,1);
b = randn(1,1);


alpha = 0.005;
N_epochs = 50
for epoch = 1:N_epochs
    shuffle_idx = randperm(N*2);
    for i = 1:((2*N)/10)
       idx = ((i-1)*10 + 1):(i*10);

       y_bs = y(shuffle_idx(idx));
       x_bs = X(shuffle_idx(idx),:);


       z = max(0,y_bs .* (x_bs * w + b));


       w_gradients= -y_bs.*x_bs;
       w_gradients(z>0,:) = 0;
       w_gradients = sum(w_gradients,1)' * 1/size(x_bs,1);

       b_gradients = -y_bs;
       b_gradients(z>0,:) = 0;
       b_gradients = sum(b_gradients)' * 1/size(x_bs,1);

       w = w - alpha*w_gradients;
       b = b - alpha*b_gradients;

       predictions = (((X*w + b) >= 0) - .5)*2';
       acc = mean(predictions == y)
       
       mesh_preds = X_mesh*w+b;
       [~,h] = contour(X1,X2,reshape(X_mesh*w+b,size(X1)),[-0.1*min(mesh_preds),0,.1*max(mesh_preds)]);
       pause(0.001)
       delete(h);
       
    end
end

