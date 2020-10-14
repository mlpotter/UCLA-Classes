close all; clear all; clc; 
n = 3;

for r = 1:n
    urn(r,:) = [r-1,n-r];
end

trials = 10000;
count1 = 0;
count2 = 0;
count3 = 0;
for i = 1:trials
    pick_urn = randperm(n,1);
    pick_balls = randperm(2,2);
    urn_contents = urn(pick_urn,:);
    
    w = urn_contents/sum(urn_contents);
    choice1 = randsample([1,2],1,true,w);
    urn_contents(choice1) = urn_contents(choice1)-1;
    
    w = urn_contents/sum(urn_contents);
    choice2 = randsample([1,2],1,true,w);
    
    if choice2 == 2
        count1 = count1 + 1;
    end
    
    if (choice2 == 2) & (choice1 == 2)
       count2 = count2 + 1;
    end
    
    if (choice1 == 2)
       count3 = count3 + 1; 
    end
end
%%
sum1 = 0;
for r = 1:n
    sum1 = sum1 + 1/n * ( ((r-1)/(n - 1)) * (n-r)/(n-1-1) + ((n-r)/(n - 1))*(n-r-1)/(n-1-1) );
end

fprintf("A. Experimental %.4f\n",count1/trials)
fprintf("A. Theoretical %.4f\n",sum1)

sum2 = 0;
sum3 = 0;
for r = 1:n
    sum2 = sum2 + (1/n * (n-r)/(n - 1) * (n-r-1)/(n-1-1));
    sum3 = sum3 + (1/n * (n-r)/(n-1));
end

fprintf("B. Experimental %.4f\n",count2/count3)
fprintf("B. Theoretical %.4f\n",sum2/sum3)