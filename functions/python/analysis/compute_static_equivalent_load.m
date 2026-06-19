function [static_equivalent_load] = compute_static_equivalent_load( fn , design_vector )

	global problem;

	if nargin == 0
		disp('Must give a file name to save the result')
		error();
	end

	if nargin == 1

		%% clear everything
		close all; clc;

		%% Add paths
		addpath(genpath('opt'),genpath('data'),genpath('plot'),genpath('analysis'),genpath('validation'));

		%% Define input parameters
		problem = define_input_tower_monopile;

		% set a design data structure
		design_vector = problem.initial_vector;

	end

	x_dimless = expand_design_vector(problem, design_vector);
	N=problem.info.num_sec;
	design_dimless.D=x_dimless(1:N);
	design_dimless.t=x_dimless(N+1:2*N);
	design = scale2physical(design_dimless, problem.scaling);

	%%% If you want to carry out the time integration on an arbitrary design, then simply pass a different design object at this point

	%Cross section properties
	[ csprops ] = compute_csprops( problem, design );

	%Build cross section stiffness and mass matrix
	[ constitutive ] = compute_constitutive( problem, csprops );

	%element stiffness and mass matrix
	[ beam.Ke, beam.Me ] = compute_KeAndMe( problem, constitutive );

	ndof = (problem.info.num_sec*3+1)*6;
	x0 = zeros(ndof, 1);
	v0 = zeros(ndof, 1);
	a0 = zeros(ndof, 1);

	% carry out the time integration
	disp('> Integrating in time')
	gamma=problem.info.newmarkBete_gamma; % 0.51 to give a little numerical damping
	beta=problem.info.newmarkBete_beta; % known relationship for better stability
	t_final=problem.info.time_final;
	time_step=problem.info.time_step;
	[ static_equivalent_load.times , static_equivalent_load.x, static_equivalent_load.v, static_equivalent_load.a, static_equivalent_load.f ] = compute_time_integration( problem, design, constitutive , 0.0, t_final, time_step, x0, v0, a0, gamma, beta, 1.0);
	%                                                                   start_time, end_time, time step,  NB-gamma, NB-beta, relaxation-factor

	% get the number of time steps
	n_time_steps = size(static_equivalent_load.x,2);

	% calculate the cross section loads for the whole problem
	disp('> Calculating time series of internal forces and moments')
	static_equivalent_load.cxf=zeros(problem.frans.mdim_1d,problem.frans.ne_1d,n_time_steps);
	for atT=1:n_time_steps
		[ static_equivalent_load.cxf(:,:,atT) ]=FRANS_RecoverForcesAndMomentsAtReferencePoint_withElm(beam.Ke,beam.Me,static_equivalent_load.x(:,atT),problem.frans);
	end

	% calculate the stress history
	disp('> Calculating time series of stress')
	static_equivalent_load.stress_in_time=zeros(2,problem.frans.ne_1d,n_time_steps);
	Ix = csprops.total.areaIx; %Bending stiffness around x axis
	Iy = csprops.total.areaIy; %Bending stiffness around y axis
	H=design.D;
	W=design.D;
	for atT=1:n_time_steps
		static_equivalent_load.stress_in_time(1,:,atT) = static_equivalent_load.cxf(4,:,atT)'.*H.*0.5./Ix;
		static_equivalent_load.stress_in_time(2,:,atT) = static_equivalent_load.cxf(5,:,atT)'.*W.*0.5./Iy;
	end

	% carry out the rainflow counting
	disp('> Calculating fatigue cycle counts');
	static_equivalent_load.cycle_counts=cell(2,problem.frans.ne_1d);
	for atElm=1:problem.frans.ne_1d
		y_stress = reshape(static_equivalent_load.stress_in_time(1,atElm,:),[],1);
		x_stress = reshape(static_equivalent_load.stress_in_time(2,atElm,:),[],1);
		static_equivalent_load.cycle_counts{1,atElm}=rainflow(y_stress);
		static_equivalent_load.cycle_counts{2,atElm}=rainflow(x_stress);
	end

	% carry out the damage calculation (note damage is the same as UR)
	disp('> Calculating the damage');
	static_equivalent_load.damage=zeros(2,problem.frans.ne_1d);
	%Sut=problem.materialdata(1).Sut;
	m=problem.materialdata(1).SNSlope;
	a=problem.materialdata(1).SNIntercept;
	eLim=problem.materialdata(1).EnduranceLimit;
	fatigue_safety_factor=1.2;
	% 20years, 365.25days, 24hours, 60minutes, 60seconds / simulation time in seconds
	D_scaling=20.0*365.25*24.0*60.0*60.0/t_final;
	for atElm=1:problem.frans.ne_1d
		for atDir=1:2
			D=0.0;
			C=static_equivalent_load.cycle_counts{atDir,atElm};
			for atC=1:size(C,1)
				% apply the goodman correction
				Sar=C(atC,2)*0.5/(1.0-(C(atC,3)/a));
				Sar=Sar*fatigue_safety_factor;
				if Sar>eLim
					% Number of cycles for the limit
					N=((Sar)/a)^(-m);
					% add the damage for that cycle cuont
					D=D+C(atC,1)/N;
				end
			end
			static_equivalent_load.damage(atDir,atElm)=D*D_scaling;
		end
	end

	% carry out the damage calculation (note damage is the same as UR)
	disp('> Calculating the equivalent stress and moments');
	% This is effectively what the equivalent stress, assuming that the stress limit is 1.0
	% The ^(1/m) accounts for the fact that the damage increases by a power of 4 as stress increases
	static_equivalent_load.stress_eq=static_equivalent_load.damage.^(1.0/(m*2.0));
	%%% Calculate the equivalent moment that would produce the target stress
	static_equivalent_load.M_eq(1,:)=static_equivalent_load.stress_eq(1,:).*2.0.*Ix'./H';
	static_equivalent_load.M_eq(2,:)=static_equivalent_load.stress_eq(2,:).*2.0.*Iy'./W';

	% carry out the damage calculation (note damage is the same as UR)
	disp('> Equivalent force');
	static_equivalent_load.F_eq = zeros(problem.frans.ne_1d, 2);
	DeltaZMtx = zeros(problem.frans.ne_1d);
	for i=1:problem.frans.ne_1d
		for j=i:problem.frans.ne_1d
			DeltaZMtx(i,j)=problem.info.z_nodes(j+1)-problem.info.z_nodes(i);
		end
	end
	% The following is the equivalent force that one would need to apply, such that a stress in bending equal to 1 would represent the structure at the limit in terms of fatigue
	static_equivalent_load.F_eq = DeltaZMtx\static_equivalent_load.M_eq';

	% Save the figure
	disp('> saving the static_equivalent_load data structure');
	save(fn, 'static_equivalent_load');

	% collect the maximum damage to print the result
	max_damage=max(max(static_equivalent_load.damage));
	str=sprintf('Maximum Damage: %f', max_damage);
	disp(str);

end

