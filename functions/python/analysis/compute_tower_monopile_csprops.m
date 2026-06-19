function [ csprops ] = compute_csprops( problem, design )
% Analyze the cross section properties

	%% Parameters
	utils.E = problem.materialdata(1).E1;
	utils.G = problem.materialdata(1).G12;
	utils.rho = problem.materialdata(1).rho;
	utils.l = problem.info.delta_z;
	utils.N = problem.info.num_sec;

	%% get the design variables
	D=design.D;
	t=design.t;

	%% Values
	%area
	csprops.A = pi*( D.*t - t.^2.0 ) ;	% beam area [m^2]
	csprops.areaIx = pi*( D.^3.0.*t - 3.0.*D.^2.0.*t.^2.0 + 4.0.*D.*t.^3.0 - 2.0.*t.^4.0 ) / 8.0 ;
	csprops.areaIy = csprops.areaIx;	% second moments of area [m^4]
	csprops.K = 2.0*csprops.areaIx;
	%stiffness
	csprops.EA = utils.E * csprops.A;
	csprops.kGA = 2.0*csprops.A * utils.G; %Adding some artifically high shear stiffness to avoid zero shear stiffness
	csprops.EIx = utils.E*csprops.areaIx;
	csprops.EIy = utils.E*csprops.areaIy;	% second moments of area [m^4]
	csprops.GK = utils.G*csprops.K;	% torsion constant [m^4]
	%mass
	csprops.ms = csprops.A .* utils.rho; % mass per unit length
	csprops.me = utils.l * csprops.ms; %element mass
	csprops.Ix = utils.rho.*csprops.areaIx;
	csprops.Iy = utils.rho.*csprops.areaIy;

	%% Gradients
	%area
	csprops.grad{1}.A = pi*t;
	csprops.grad{2}.A = pi*( D - 2.0.*t );
	csprops.grad{1}.areaIx = pi*( 3.0.*D.^2.0.*t - 6.0.*D.*t.^2.0 + 4.0.*t.^3.0 )/8.0;
	csprops.grad{2}.areaIx = pi*( D.^3.0 - 6.0.*D.^2.0.*t + 12.0.*D.*t.^2.0 - 8.0.*t.^3.0 )/8.0;
	csprops.grad{1}.areaIy = csprops.grad{1}.areaIx;
	csprops.grad{2}.areaIy = csprops.grad{2}.areaIx;
	csprops.grad{1}.K = 2.0*csprops.grad{1}.areaIx;
	csprops.grad{2}.K = 2.0*csprops.grad{2}.areaIx;
	csprops.grad{1}.EA = utils.E * csprops.grad{1}.A;
	csprops.grad{2}.EA = utils.E * csprops.grad{2}.A;
	csprops.grad{1}.kGA = 2.0*utils.G * csprops.grad{1}.A;
	csprops.grad{2}.kGA = 2.0*utils.G * csprops.grad{2}.A;
	csprops.grad{1}.EIx = utils.E * csprops.grad{1}.areaIx;
	csprops.grad{2}.EIx = utils.E * csprops.grad{2}.areaIx;
	csprops.grad{1}.EIy = utils.E * csprops.grad{1}.areaIy;
	csprops.grad{2}.EIy = utils.E * csprops.grad{2}.areaIy;
	csprops.grad{1}.GK = utils.G * csprops.grad{1}.K;
	csprops.grad{2}.GK = utils.G * csprops.grad{2}.K;
	csprops.grad{1}.ms = utils.rho * csprops.grad{1}.A;
	csprops.grad{2}.ms = utils.rho * csprops.grad{2}.A;
	csprops.grad{1}.me = utils.l * csprops.grad{1}.ms;
	csprops.grad{2}.me = utils.l * csprops.grad{2}.ms;
	csprops.grad{1}.Ix = utils.rho.* csprops.grad{1}.areaIx;
	csprops.grad{2}.Ix = utils.rho.* csprops.grad{2}.areaIx;
	csprops.grad{1}.Iy = utils.rho.* csprops.grad{1}.areaIy;
	csprops.grad{2}.Iy = utils.rho.* csprops.grad{2}.areaIy;

	%%global fd_data;
	%%if fd_data.state==1
	%%	fd_data.data= [ diag(csprops.grad{1}.areaIx) ; diag(csprops.grad{2}.areaIx) ];
	%%elseif fd_data.state==2
	%%	fd_data.data=csprops.areaIx';
	%%elseif fd_data.state==3
	%%	fd_data.data=csprops.areaIx';
	%%end
end

