function [ load ] = compute_steadyLoad( problem , design )

	% sizes and initialization
	N=problem.info.num_sec;
	Ndof=(1+3*N)*6;
	load=zeros(Ndof,1);

	% Calculate the rotor thrust
	rho_air = problem.info.rho_air;
	D = problem.info.rotor_diameter;
	V = problem.info.wind_speed;
	CT = problem.info.CT;
	A = 0.25*pi*D^2.0;
	thrust = 0.5*rho_air*A*CT*V^2.0;

	% Calculate rotor torque
	%P = problem.info.power/problem.info.eff;
	%tsr = problem.info.tip_speed_ratio;
	%omega=2.0*tsr*V/D;
	%torque=P/omega;

	% Add tower top forces to the problem
	load(end-4)=thrust;
	%load(end-1)=torque;

	% These are the wave parameters
	water_nNode=problem.info.water_nNode;
	water_dofs=problem.info.water_dofs;
	water_elms=problem.info.water_elms;
	Ucurr=problem.info.Ucurr;
	delta_z=problem.info.delta_z;
	rho_water=problem.info.rho_water;
	CD=problem.info.CD;

	% loop through all the water elements
	for atElm=1:water_nNode

		% extract the id's
		dof_low=water_dofs(atElm)+2;
		dof_hgh=water_dofs(atElm+1)+2;
		elm=water_elms(atElm);

		% calculate the force values
		U0=Ucurr(atElm);
		sgn=1.0;
		if U0<0.0
			sgn=-1.0;
		end
		elmFperL_drag=0.5*rho_water*CD.*design.D(elm).*U0^2*sgn;
		elmF=0.5*delta_z*elmFperL_drag;
		load(dof_low)=load(dof_low)+elmF;
		load(dof_hgh)=load(dof_hgh)+elmF;

	end

end
