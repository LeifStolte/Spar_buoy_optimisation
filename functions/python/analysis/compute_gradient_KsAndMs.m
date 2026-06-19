function [ grad ] = compute_gradient_KsAndMs( problem, csprops )
%Compute gradients of cross section stiffness and mass matrix

grad = cell(problem.nvar,1);
for i=1:problem.nvar
	grad{i}=struct('Ks',[],'Ms',[]);
end

%grad = {struct([]),struct([]),struct([])}
N = problem.info.num_sec;
for grd=1:problem.nvar

	grad{grd}.Ks = zeros(6,6,N);
	grad{grd}.Ms = zeros(6,6,N);

	%% Gradients 
	%Using compute_Ks to assemble dKsdx by simply providing the gradients of
	%the cross section stiffness properties as input
	csprops_aux = csprops.total.grad{grd};
	for s = 1:N
		[ grad{grd}.Ks(:,:,s) ] = compute_Ks( csprops_aux.kGA(s), csprops_aux.EA(s), ...
			csprops_aux.EIx(s), csprops_aux.EIy(s), csprops_aux.GK(s) );
		[ grad{grd}.Ms(:,:,s) ] = compute_Ms( csprops_aux.ms(s), csprops_aux.Ix(s), ...
			csprops_aux.Iy(s) );
	end

end

end

