function [ u_tip, dudx_tip, du ] = compute_tip_displacement( problem, beam, solution )

%% Compute tip displacement constraint
N = problem.info.num_sec;
edof = problem.frans.edof_1d; %element degrees of freedom, each column is an element
displacement = solution.u(:,1); %Using displacements from flapwise load case

%Value
%dof for tip displacement uy
nn_tip = problem.frans.nn_1d;
dof_tip = (nn_tip-1)*6+2;

%Extract tip displacement
u_tip = displacement(dof_tip); 

% copy the data into the ouput data structure
for grd=1:problem.nvar
 	du.grad{grd} = solution.grad{grd}.u(:,1,:);
	dudx_tip.grad{grd}=zeros(N,1);
	dudx_tip.grad{grd}(:) = solution.grad{grd}.u(dof_tip,1,:);
end

%% %Gradient
%% %First we determine dKdx*u by multiplying dKdx with u but elementwise, i.e., dKedx*ue because each
%% %design variable affects only one element. The resulting column vector is
%% %mapped onto dKdxu which has the size of K.
%% dKdxu = zeros(size(beam.K,1),N*3);
%% for e = 1:N
%%     ue = displacement(edof(:,e));
%% 	for grd=1:problem.nvar
%% 		dKdxu(edof(:,e),e+(grd-1)*N) = beam.grad{grd}.Ke(:,:,e) * ue;
%% 	end
%% end
%% 
%% % This is a modification that only works for cantilever beams where the
%% % first six degrees of freedom at fixed to zero. The corresponding
%% % deriviatives must also be set to zero.
%% dKdxu(1:6,:) = 0;
%% 
%% %Determine gradient of displacements
%% dudx = - beam.K \ dKdxu;
%% 
%% %Storing gradient values for output
%% for grd=1:problem.nvar
%% 	du.grad{grd} = dudx(:,((grd-1)*N+1):(grd*N));
%% end
%% 
%% %Extract tip displacement gradients
%% for grd=1:problem.nvar
%% 	dudx_tip.grad{grd} = dudx(dof_tip,((grd-1)*N+1):(grd*N))';
%% end

end
