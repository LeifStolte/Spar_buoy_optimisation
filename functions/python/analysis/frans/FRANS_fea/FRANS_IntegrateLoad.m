function [Npres]=FRANS_IntegrateLoad(enum,utils)

Npres=zeros(6,utils.mdim_1d);
%Number of Gauss points
gpoints=utils.nnpe_1d;
%Start integration
for v=1:gpoints     %Iterate gauss points along the length
    xx=utils.GQ(v,2,gpoints-1);    %X position of Gauss point
    wxx=utils.GQ(v,1,gpoints-1);    %Weight from Gauss quadrature
    %Evaluate Jacobian
    [detJ]=FRANS_B4L_Jacobian(xx,enum,utils);
    %Evaluate shape functions
    [N]=FRANS_B4L_ShapeFunctions(xx);
    %Element stiffness matrix
    Npres=Npres+wxx*detJ*N;
end
Npres=Npres';

end