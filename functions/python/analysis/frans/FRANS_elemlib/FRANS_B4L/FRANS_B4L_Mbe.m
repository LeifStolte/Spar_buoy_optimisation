function [ Me ] = FRANS_B4L_Mbe( Ms, N, detJ )
%B4L_MBE Summary of this function goes here
%   Detailed explanation goes here

Me=N'*Ms*N*detJ;

end

