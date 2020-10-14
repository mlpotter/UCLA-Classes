close all; clear all; clc;
index = 0;
for i = 1:6
    for j = 1:6
        index = index + 1;
        S(index,:) = [i,j];
    end
end
%%
PA = mean(((S(:,1) == 1) | (S(:,1) == 2) | (S(:,1) == 3)));
PB = mean(((S(:,1) == 2) | (S(:,1) == 3) | (S(:,1) == 6)));
PC = mean(sum(S,2)==9);
%%
PAB = mean(((S(:,1) == 1) | (S(:,1) == 2) | (S(:,1) == 3)) & ((S(:,1) == 2) | (S(:,1) == 3) | (S(:,1) == 6)));
PAC = mean(((S(:,1) == 1) | (S(:,1) == 2) | (S(:,1) == 3)) & (sum(S,2)==9));
PBC = mean(((S(:,1) == 2) | (S(:,1) == 3) | (S(:,1) == 6)) & (sum(S,2)==9));
PABC = mean(((S(:,1) == 1) | (S(:,1) == 2) | (S(:,1) == 3)) & ((S(:,1) == 2) | (S(:,1) == 3) | (S(:,1) == 6)) & (sum(S,2)==9));

fprintf("PA=%.5f\n",PA)
fprintf("PB=%.5f\n",PB)
fprintf("PC=%.5f\n",PC)
fprintf("PAB=%.5f - PAPB=%.5f\n",PAB,PA*PB)
fprintf("PAC=%.5f - PAPC=%.5f\n",PAC,PA*PC)
fprintf("PBC=%.5f - PBPC=%.5f\n",PBC,PB*PC)
fprintf("PABC=%.5f - PAPBPC=%.5f\n",PABC,PA*PB*PC)