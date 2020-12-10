n = 49
mu_ = 205;
std_ = 15;

std_w = sqrt(n*std_^2)
mu_w = n*mu_;

1-qfunc((9800-mu_w)/std_w)
%%
W = sum(normrnd(mu_,std_,1000000,n),2);
mean(W <= 9800)