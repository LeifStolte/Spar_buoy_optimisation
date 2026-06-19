function [ utils ] = FRANS_Utils( options )

%Print header 
FRANS_PrintHeader

fprintf(1,'> Started building working arrays \n');

%Storing input data
if isfield(options,'L')
    %Number of elements
    utils.ne_1d=options.ne;
    %Number of nodes;
    utils.nn_1d=utils.ne_1d*3+1;
    %Element connectivity table
    utils.el_1d=[(1:utils.ne_1d);
        reshape((1:3*utils.ne_1d)',3,[]);
        4:3:3*utils.ne_1d+1]';
    %Nodal list
    utils.nl_1d=[1:3*utils.ne_1d+1;
        0:options.L/(utils.nn_1d-1):options.L;
        zeros(2,3*utils.ne_1d+1)]';
    %Boundary conditions
    utils.bc_1d=[ones(6,1) (1:6)' zeros(6,1)];
    %Force
    utils.f_1d=[options.f(:,1:2) ...
                min([options.f(:,3) ones(size(options.f,1))*utils.nn_1d],[],2)...
                options.f(:,4:end)];
else
    %Manage folder names
    originalFolder = pwd;
    utils.foldername=options.foldername;
    cd (utils.foldername)
    utils.nl_1d=importdata('N1D.in');
    utils.el_1d=importdata('E1D.in');
    utils.bc_1d=importdata('BC1D.in');
    utils.f_1d=importdata('F1D.in');
    cd(originalFolder)
    %Constants
    utils.ne_1d=size(utils.el_1d,1); %Number of elements
    utils.nn_1d=size(utils.nl_1d,1); %Number of nodes
end

utils.nnpe_1d=4; %Number of nodes per element
utils.mdim_1d=4*6; %Number of dof per element  

%% Load 
%Arrays
utils.f_1d=utils.f_1d;
utils.bc_1d=utils.bc_1d;
%Integers
utils.nb_1d=size(utils.bc_1d,1);
utils.np_1d=size(utils.f_1d,1);
utils.nlc_1d=max(utils.f_1d(:,1));


%Store nodal positions for each element in a vector
[utils.pr_1d]=ReorderNodalPos(utils);

%Calculate local to global DOF mapping
[utils.edof_1d]=CalcEdof(utils);

%Retrieve Gauss quadrature points and weights
[utils.GQ]=FRANS_GaussQuad;

fprintf(1,'> Finished building working arrays \n');
   
    function [edof_1d]=CalcEdof(utils)
        
        %1D - Mapping from local to global DOF
%         edof_1d=zeros(utils.mdim_1d/utils.nnpe_1d*(utils.nnpe_1d-1)+utils.ne_1d,utils.ne_1d);
        for e=1:utils.ne_1d
            for jj=1:utils.nnpe_1d
                for j=1:utils.mdim_1d/utils.nnpe_1d
                    edof_1d(utils.mdim_1d/utils.nnpe_1d*(jj-1)+j,e) = utils.mdim_1d /utils.nnpe_1d * (utils.el_1d(e,jj+1)-1)+j;
                end
            end
        end
                
    end

    function [pr_1d]=ReorderNodalPos(utils)
        
        %Reorganize nodal positions - 1D
        pr_1d=zeros(utils.nnpe_1d*3,utils.ne_1d);
        for ii=1:utils.ne_1d
            for i=1:utils.nnpe_1d
                pr_1d((i-1)*3+1:(i-1)*3+3,ii)=utils.nl_1d((utils.el_1d(ii,i+1)),2:4);
            end
        end

    end

end

