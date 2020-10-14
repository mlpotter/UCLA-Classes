close all; clear all; clc;
n = 6;
count = 0;
trials = 1000000;
for trial = 1:trials
    ball_positions = ones(1,n*2);
    idx = randperm(n*2,n);
    ball_positions(idx) = 2;
    case_ = logical(1);
    for i = 1:n
    %     fprintf("%d %d\n",(i-1)*2+1,i*2)
  
        person_balls = ball_positions((i-1)*2+1 : i*2);

        case_ = case_ & (sum(person_balls) == 3);
 
    end
    if case_
        count = count + 1;
    end
end
fprintf("Actual Solution %.4f\n",count/trials)
numerator = (2^n);
denominator = factorial(2*n)/(factorial(n)*factorial(n));
fprintf("Theoretical Solution %.4f\n",numerator/denominator)
%%
close all; clear all; clc;
n = 5;
count = 0;
trials = 1000000;
for trial = 1:trials
    ball_positions = ones(1,n*2);
    ball_positions(1:n) = 2;
    idx = randperm(n*2);
    case_ = logical(1);
    for i = 1:n
    %     fprintf("%d %d\n",(i-1)*2+1,i*2)
  
        person_ball_idx = idx((i-1)*2+1 : i*2);
        person_balls = ball_positions(person_ball_idx);

        case_ = case_ & (sum(person_balls) == 3);
 
    end
    if case_
        count = count + 1;
    end
end
%%
fprintf("Actual Solution %.4f\n",count/trials)
numerator = (2^n);
denominator = factorial(2*n)/(factorial(n)*factorial(n));
fprintf("Theoretical Solution %.4f\n",numerator/denominator)

sum = 0
for m = 1:floor(n/2)
    sum = sum + factorial(n)/(factorial(m)*factorial(m)*factorial(n-2*m))
end
1/sum
numerator/denominator