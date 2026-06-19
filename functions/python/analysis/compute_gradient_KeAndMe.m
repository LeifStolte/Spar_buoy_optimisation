function [ grad ] = compute_gradient_KeAndMe( problem, constitutive )
%Gradients of element stiffness and mass matrix
%Note that the function FRANS_B4L which was used to determine the element
%stiffness and mass matrix is now used to compute the gradients of the
%element stiffness and mass matrix by simply giving dKsdx and dMsdx as
%input.

grad = cell(problem.nvar,1);
for grd=1:problem.nvar
	grad{grd}=struct('Ke',[],'Me',[]);
end

%Aux
mdim = problem.frans.mdim_1d;
N = problem.info.num_sec;

%Gradients
for grd=1:problem.nvar
	grad{grd}.Ke = zeros(mdim, mdim, N);
	grad{grd}.Me = zeros(mdim, mdim, N);
end

for e=1:N
	for grd=1:problem.nvar
		%dKeda and dMeda
		[ grad{grd}.Ke(:,:,e), grad{grd}.Me(:,:,e) ] = FRANS_B4L( e, constitutive.grad{grd}.Ks(:,:,e),...
			constitutive.grad{grd}.Ms(:,:,e), problem.frans );
	end
end

end
