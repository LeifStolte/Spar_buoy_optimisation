function [ eigfreq, eigvec ] = FRANS_SolveModal( Kbg, Mbg )
%********************************************************
% File: SolveModal1D.m
%   Solve eigenvalue problem associated with the determination of the
%   eigenfrequencies and eigenvectors of the beam finite element assembly.
% Syntax:
%   [ eigfreq, eigvec ] = SolveModal1D( Kbg, Mbg )
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

%Solve eigenvalue problem to determine the first 20 frequencies
opts = [];
opts.tol = 1.0e-13;
opts.v0 = ones(size(Kbg,1),1);
[eigvec,eigfreq_m]=eigs(Kbg,Mbg,20,'sm',opts);

% Mass-normalize the eigenvectors;
a=sqrt(diag(eigvec'*full(Mbg)*eigvec));
for i=1:size(a,1)
    eigvec(:,i)=eigvec(:,i)/a(i);
end

%Sorting eigenfrequencies in ascending order of magnitude
[eigfreq,indx]=sort(diag(eigfreq_m),'ascend');

%Sorting eigenvectors
eigvec=eigvec(:,indx);

%Correcting eigenvectors to include zeros where the constraints should be
%CORRECT FOR GENERAL CASE - ONLY WORKING FOR CANTILEVER BEAM
eigvec=[zeros(6,size(eigvec,2)) ; eigvec];

end

