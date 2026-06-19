function [ Kbg, Mbg ]=FRANS_Assemble( utils, Ks, Ms )
%********************************************************
% File: FRANS_Assemble.m
%   Assemble global beam finite element stiffness and mass matrix
%   in sparse format
% Syntax:
%   [ Kbg, Mbg ]=FRANS_Assemble( utils, Ks, Ms )
% Input:
%   utils   :  Structure with input data and useful arrays
%   Ks      :  Array of cross section stiffness matrices. For prismatic beams
%              Ks is (6 x 6 x 1), i.e., same Ks for all beam FE elements. 
%              For beams with varying cross section along the legnth Ks is 
%              (6 x 6 x nb), where nb is the number of beam FE elements.
%   Ms      :  Array of cross section mass matrices. Same rule as for Ks.
% Output:
%  Kbg      :  Global beam finite element stiffness matrix in sparse format
%  Mbg      :  Global beam finite element mass matrix in sparse format
%
% Date:
%   Version 1.0    07.02.2012
%
% (c) DTU Wind Energy
%********************************************************

%% Initialize arrays
nKb=(utils.nnpe_1d*6)*(utils.nnpe_1d*6);
iK=zeros(utils.ne_1d*nKb,1);
jK=zeros(utils.ne_1d*nKb,1);
vK=zeros(utils.ne_1d*nKb,1);
vM=zeros(utils.ne_1d*nKb,1);
Ksmat=zeros(6,6,utils.ne_1d);
Msmat=zeros(6,6,utils.ne_1d);

%% Check for size of Ks and Ms matrices
if(size(Ks,3) == utils.ne_1d || size(Ms,3) == utils.ne_1d )
    Ksmat=Ks;
    Msmat=Ms;
elseif(size(Ks,3) == 1 || size(Ms,3) == 1 )
    for i=1:utils.ne_1d
        Ksmat(:,:,i)=Ks;
        Msmat(:,:,i)=Ms;
    end
else
    fprintf(1,'Error in FRANS_Assemble: Number of Ks and Ms matrices does not match number of beam elements! \n');
    return
end

%Loop beam finite elements
for e=1:utils.ne_1d
    %Evaluate element stiffness matrix
    [Ke,Me]=FRANS_B4L(e,Ksmat(:,:,e),Msmat(:,:,e),utils);
    %Assemble global matrices
    iK((e-1)*nKb+1:e*nKb)=reshape(kron(utils.edof_1d(:,e),ones(size(Ke,1),1)),size(Ke,1)*size(Ke,2),1);
    jK((e-1)*nKb+1:e*nKb)=reshape(kron(utils.edof_1d(:,e),ones(1,size(Ke,1))),size(Ke,1)*size(Ke,2),1);
    vK((e-1)*nKb+1:e*nKb)=reshape(Ke,size(Ke,1)*size(Ke,2),1);
    vM((e-1)*nKb+1:e*nKb)=reshape(Me,size(Me,1)*size(Me,2),1);
end

%Assemble global matrices
Kbg=sparse(iK,jK,vK);
Mbg=sparse(iK,jK,vM);

% Make sure that the stiffness matrix is numerically symmetric
Kbg = (Kbg + Kbg')/2;

end

