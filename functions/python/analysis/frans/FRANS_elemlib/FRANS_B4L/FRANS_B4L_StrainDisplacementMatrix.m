function [ B ] = FRANS_B4L_StrainDisplacementMatrix( detJ, N, Nprime )
%B4L_ Summary of this function goes here
%   Detailed explanation goes here

%B0 and B1 MATRICES
B0=zeros(6);
B0(1,5)=-1.;
B0(2,4)=1.;
B1=eye(6);

%Strain displacement matrix
B=B0*N+1/detJ*B1*Nprime;

end

