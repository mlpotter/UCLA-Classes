close all; clear all; clc;
index = 0;
for i = 1:2
    for j = 1:2
        index = index + 1;
        S(index,:) = [i,j];
    end
end
%% 1 is head, 2 is tails
PA = mean( (S(:,1) == 1) );
PB = mean( (S(:,2) == 1) );
PC = mean( (S(:,1) == S(:,2)) );

PAB = mean( (S(:,1) == 1) &  (S(:,2) == 1) );
PAC = mean( (S(:,1) == 1) &  (S(:,1) == S(:,2)) );
PBC = mean( (S(:,2) == 1) &  (S(:,1) == S(:,2)) );
PABC = mean( (S(:,1) == 1) &  (S(:,2) == 1) & (S(:,1) == S(:,2)) );

fprintf("PA=%.5f\n",PA)
fprintf("PB=%.5f\n",PB)
fprintf("PC=%.5f\n",PC)
fprintf("PAB=%.5f - PAPB=%.5f\n",PAB,PA*PB)
fprintf("PAC=%.5f - PAPC=%.5f\n",PAC,PA*PC)
fprintf("PBC=%.5f - PBPC=%.5f\n",PBC,PB*PC)
fprintf("PABC=%.5f - PAPBPC=%.5f\n",PABC,PA*PB*PC)