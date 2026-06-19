function [ Kbe, Mbe ] = FRANS_B4L( enum, Ks, Ms, utils )
%B4L_ Summary of this function goes here
%   Detailed explanation goes here

Kbe=zeros(24,24);
Mbe=zeros(24,24);

%Number of Gauss points
gpoints=4;
%Start integration
for i=1:gpoints     %Iterate gauss points along the length
    xx=utils.GQ(i,2,gpoints-1);    %X position of Gauss point
    wxx=utils.GQ(i,1,gpoints-1);    %Weight from Gauss quadrature
    %Evaluate Jacobian
    [detJ]=FRANS_B4L_Jacobian(xx,enum,utils);
    %Evaluate shape functions
    [N,Nprime]=FRANS_B4L_ShapeFunctions(xx);
    %Element stiffness matrix
    [Kee]=FRANS_B4L_Kbe(Ks,detJ,N,Nprime);
    Kbe=Kbe+Kee*wxx;
    %Element mass matrix
    [Mee]=FRANS_B4L_Mbe(Ms,N,detJ);
    Mbe=Mbe+Mee*wxx;
end
            
end

