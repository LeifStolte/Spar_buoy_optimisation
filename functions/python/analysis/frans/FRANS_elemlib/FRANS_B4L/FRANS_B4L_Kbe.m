function [ Ke ] = FRANS_B4L_Kbe( Ks, detJ, N, Nprime)
%B4L_KBE Summary of this function goes here
%   Detailed explanation goes here

%Evaluate the strain displacement matrix
[ B ]=FRANS_B4L_StrainDisplacementMatrix(detJ, N, Nprime);

%Build element stiffness matrix
Ke=B'*Ks*B*detJ;

end

