function [K,M]=FRANS_EigenEnforce(K,M,utils)
%********************************************************
% File: FRANS_EigenEnforce.m
%   Solve eigenvalue problem associated with the determination of the
%   eigenfrequencies and eigenvectors of the beam finite element assembly.
% Syntax:
%   [ eigfreq, eigvec ] = FRANS_EigenEnforce( Kbg, Mbg )
% Input:
%   Kbg     :  Global beam finite element stiffness matrix
%   Mbg     :  Global beam finite element mass matrix
% Output:
%   eigfreq :  Column vector of eigenfrequencies in ascencding order
%   eigvec  :  Matrix of mass-normalized eigenvectors ordered according
%              to the eigenfrequencies
%
% Date:
%   Version 1.0    07.02.2012
%
% (c) DTU Wind Energy
%********************************************************

%Enforce boundary conditions on stiffness matrix
for i = 1:utils.nb_1d
    idof(i) = (utils.mdim_1d/utils.nnpe_1d)*(utils.bc_1d(i,1)-1) + utils.bc_1d(i,2);
end
K(:, idof) = [];
K(idof, :) = [];
M(:, idof) = [];
M(idof,:) = [];
end
