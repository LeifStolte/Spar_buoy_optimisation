function [ times , x, v, a, f ] = compute_time_integration( problem, design, constitutive , t0, tf, dt, x0, v0, a0, gamma, beta, relax)

	r_norm_target=problem.info.r_norm_target;
	iter_limit=problem.info.iter_limit;

	% Retrieve the steady state properties
	load = compute_steadyLoad( problem , design);

	% get the beam properties
	beam = compute_beam( problem, design, constitutive, load);
	Mdyn=beam.Mdyn;
	Kdyn=beam.Kdyn;
	Cdyn = Mdyn*problem.info.rayleigh_damping_mass+Kdyn*problem.info.rayleigh_damping_stiff;

	% Compute the steady state solution
	[ solution ] = compute_beam_solution( problem, beam, constitutive );

	% define some constants
	fromDeltaXToDeltaA = 1.0/(dt*dt*beta);
	fromDeltaXToDeltaV = gamma/(dt*beta);

	% initialize some of the data structures
	times=[t0];
	x=[solution.u(:,1)];
	v=[v0];
	a=[a0];

	% set the integration data
	t=t0;
	curr_x=solution.u(:,1);
	curr_v=v0;
	curr_a=a0;

	% collect the initial force
	[ curr_f , dfdx, dfdv, dfda ] = compute_unsteady_force( problem, design, 0, t , curr_x , curr_v , curr_a );

	% save the initial force
	f=[curr_f];

	% set some numerical constants
	t_print = t0;
	max_iter = 0;
	max_r = 0.0;
	max_u = 0.0;
	max_v = 0.0;
	max_a = 0.0;
	max_f = 0.0;
	step_cnt = 0;
	print_at_iter = 600.0;
	emergency_print = 1;

	while t < tf

		% advance time step
		t=t+dt;
		step_cnt=step_cnt+1;

		% save the past state
		past_x=curr_x;
		past_v=curr_v;
		past_a=curr_a;

		% predictor step
		curr_v=past_v+(1-gamma)*dt*past_a+gamma*dt*curr_a;
		curr_x=past_x+dt*past_v+(dt*dt/2.0)*((1.0-2.0*beta)*past_a+2.0*beta*curr_a);

		iter = 0;
		loop = 1;
		while loop == 1

			% compute the forces
			[ curr_f , dfdx, dfdv, dfda ] = compute_unsteady_force( problem, design, step_cnt, t , curr_x , curr_v , curr_a );

			% compute the residual
			r = Mdyn*curr_a+Cdyn*curr_v+Kdyn*curr_x-curr_f;
			% enforce boundary conditions
			r(1:6)=zeros(6,1);
			r_norm = norm(r);

			if iter>iter_limit | r_norm<r_norm_target
				loop=0;
			else
				% compute the jacobian
				J = (Mdyn-dfda)*fromDeltaXToDeltaA + (Cdyn-dfdv)*fromDeltaXToDeltaV + Kdyn - dfdx;
				% compute delta u
				delta_u=-J\r;
				% enforce boundary conditions
				delta_u(1:6)=zeros(6,1);
				% update state
				curr_a=curr_a+delta_u*relax*fromDeltaXToDeltaA;
				curr_v=curr_v+delta_u*relax*fromDeltaXToDeltaV;
				curr_x=curr_x+delta_u*relax;
				iter=iter+1;
			end
		end

		u_norm=norm(curr_x);
		v_norm=norm(curr_v);
		a_norm=norm(curr_a);
		f_norm=norm(curr_f);

		if iter>max_iter
			max_iter = iter;
		end
		if r_norm>max_r
			max_r = r_norm;
		end
		if u_norm>max_u
			max_u = u_norm;
		end
		if v_norm>max_v
			max_v = v_norm;
		end
		if a_norm>max_a
			max_a = a_norm;
		end
		if f_norm>max_f
			max_f = f_norm;
		end

		times(end+1)=t;
		x(:,end+1)=curr_x;
		v(:,end+1)=curr_v;
		a(:,end+1)=curr_a;
		f(:,end+1)=curr_f;

		if (problem.options.print_time_integration == 1 & ((((t-t_print)/dt)>print_at_iter) | t >= tf)) | (( max_iter>iter_limit | r_norm>r_norm_target ) & emergency_print==1)
			txt=sprintf('Steps %i Time %f Max-Iter %i Max-R %e Max-U %e Max-V %e Max-A %e Max-F %e', step_cnt, t, max_iter, max_r, max_u, max_v, max_a, max_f);
			disp(txt);
			t_print = t_print+print_at_iter*dt;
			if ( iter_limit>1000 | r_norm>r_norm_target )
				disp('The time integration appears to show slow progress, consider tuning the numerics and re-starting')
				emergency_print = 0;
			else
				% reset the maximum values
				max_iter = 0;
				max_r = 0.0;
				max_u = 0.0;
				max_v = 0.0;
				max_a = 0.0;
				max_f = 0.0;
			end
		end

	end
end


