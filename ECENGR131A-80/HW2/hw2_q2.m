index = 0;
for i = 1:6
    for j = 1:6
        index = index + 1;
        S(index,:) = [i,j];
    end
end

X = abs(S(:,1) - S(:,2));
fprintf("Experimental %.5f\n",mean(X >= S(:,1)))
%%
pr = 0;
for i = 1:6
    for j = 1:6
        pr = pr + (abs(i-j)>=i)/(36);
    end
end
fprintf("Experimental %.5f\n",pr)
