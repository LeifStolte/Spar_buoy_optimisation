function [ f , dfdx, dfdv, dfda ] = compute_unsteady_force( problem, design, time_index, t, x, v, a )

	% Initialize the force
	% f = compute_steadyLoad( problem , design );
	N=problem.info.num_sec;
	ndof=(1+3*N)*6;
	f=zeros(ndof,1);
	dfdx=zeros(ndof,ndof);
	dfdv=zeros(ndof,ndof);
	dfda=zeros(ndof,ndof);
	% problem.info.num_sec;

	% f = 0.5*rho*A*V^2*CT
	% f = 0.5*rho*A*CT*(V+v-U)^2
	% f = 0.5*rho*A*CT*(V^2+2.0*V*v+v^2-2.0*v*U-2.0*V*U+U^2)
	% f = 0.5*rho*A*CT*(2.0*V*v+v^2-2.0*v*U-2.0*V*U+U^2)
	% df = rho*A*CT*(U-v-V)
	rho_air = problem.info.rho_air;
	D = problem.info.rotor_diameter;
	CT = problem.info.CT;
	A = 0.25*pi*D^2.0;
	U = v(ndof-4);
	Vturb = problem.info.Uwind(time_index+1);
	sgn=1.0;
	if (Vturb-U)<0.0
		sgn=-1.0;
	end
	delta_thrust = 0.5*rho_air*A*CT*(Vturb^2.0-2.0*Vturb*U+U^2.0)*sgn;
	delta_thrust_grad_v = rho_air*A*CT*(U-Vturb)*sgn;

	% add the wind force to force vector and the gradient
	f(ndof-4)=f(ndof-4)+delta_thrust;
	dfdv(ndof-4,ndof-4)=delta_thrust_grad_v;

	% These are the wave parameters
	water_nNode=problem.info.water_nNode;
	water_dofs=problem.info.water_dofs;
	water_elms=problem.info.water_elms;
	Uwave=problem.info.Uwave;
	dUwave=problem.info.dUwave;
	Ucurr=problem.info.Ucurr;
	delta_z=problem.info.delta_z;
	rho_water=problem.info.rho_water;
	CD=problem.info.CD;
	CM=problem.info.CM;

	% loop through all the water elements
	for atElm=1:water_nNode

		% extract the id's
		dof_low=water_dofs(atElm)+2;
		dof_hgh=water_dofs(atElm+1)+2;
		elm=water_elms(atElm);

		% calculate the force values
		vel=0.5*(v(dof_low)+v(dof_hgh));
		U0=problem.info.Ucurr(atElm);
		UT=Uwave(atElm,time_index+1);
		dU=dUwave(atElm,time_index+1);
		sgn=1.0;
		if (U0+UT-vel)<0.0
			sgn=-1.0;
		end
		elmFperL_drag=0.5*rho_water*CD.*design.D(elm).*(U0*U0+2.0*U0*UT+UT^2-2.0*U0*vel-2.0*UT*vel+vel^2.0)*sgn;
		elmFperL_mass=rho_water.*CM.*0.25.*pi.*design.D(elm).*design.D(elm).*dU;
		elmF=0.5*delta_z*(elmFperL_drag+elmFperL_mass);
		f(dof_low)=f(dof_low)+elmF;
		f(dof_hgh)=f(dof_hgh)+elmF;

		% calculate the gradient
		elmFperL_drag_grad_v=rho_water*CD.*design.D(elm).*sgn.*(vel-U0-UT)*0.25*delta_z;
		dfdv(dof_low,dof_low)=dfdv(dof_low,dof_low)+elmFperL_drag_grad_v;
		dfdv(dof_low,dof_hgh)=dfdv(dof_low,dof_hgh)+elmFperL_drag_grad_v;
		dfdv(dof_hgh,dof_low)=dfdv(dof_hgh,dof_low)+elmFperL_drag_grad_v;
		dfdv(dof_hgh,dof_hgh)=dfdv(dof_hgh,dof_hgh)+elmFperL_drag_grad_v;

	end

end

