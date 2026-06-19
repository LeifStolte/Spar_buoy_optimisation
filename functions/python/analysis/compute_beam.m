function [ beam ] = compute_beam( problem, design, constitutive, load)

%Assemble global beam stiffness matrix
[ beam.Kfem, beam.Mfem ] = FRANS_Assemble( problem.frans, constitutive.Ks, constitutive.Ms );

% collect additional mass and stiffness properties if needed
if strcmp(problem.type,'tower_monopile')
	% apply concentrated mass and soil stiffness
	[ beam.Mcm ] = compute_tower_monopile_concentrated_mass_matrix( problem );
	% apply the soil stiffness
	[ beam.Ksoil beam.dKsoil ] = compute_soil_stiffness_matrix( problem , design );
	beam.M=beam.Mfem+beam.Mcm;
	beam.K=beam.Kfem+beam.Ksoil;
else
	beam.K=beam.Kfem;
	beam.M=beam.Mfem;
end

% Use the defauld load vector if not given
if nargin<4
	%Build load vector
	[ beam.p ] = FRANS_BuildLoad( problem.frans );
else
	beam.p = load;
end

%Enforce boundary conditions
[ beam.K, beam.p ] = FRANS_Enforce(beam.K, beam.p, problem.frans);
beam.Kdyn = beam.K;
ndof = (problem.info.num_sec*3+1)*6;
dummy_p = zeros(ndof,2);
[ beam.Mdyn, dummy_p ] = FRANS_Enforce( beam.M, dummy_p, problem.frans );
[ beam.eigK, beam.eigM ] = FRANS_EigenEnforce( beam.K, beam.M, problem.frans );

%element stiffness and mass matrix
[ beam.Ke, beam.Me ] = compute_KeAndMe( problem, constitutive );

%Gradients of element stiffness and mass matrix
[ beam.grad ] = compute_gradient_KeAndMe( problem, constitutive );

end
