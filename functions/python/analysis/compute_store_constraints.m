function [ c, dc, ceq, dceq ] = compute_store_constraints( problem, u_tip, dudx_tip, strain, dstrain, eta, deta, w, dw )

%Aux
scaling = problem.scaling;
N = problem.info.num_sec;
nfreq = length(problem.constraints.max_freq);

% Initialize output
if strcmp(problem.type,'blade_structure')
	c = zeros(1, 1+3*N+2*nfreq);  % 1 tip disp. constraint + N cap strain constraints + N ellipse strain constraints + N buckling constraints + 1 frequency constraint
	dc = zeros(3*N, 1+3*N+2*nfreq);  % gradients of the constraints
elseif strcmp(problem.type, 'tower_monopile')
	c = zeros(1, 1+3*N+2*nfreq);  % 1 tip disp. constraint + N cap strain constraints + N ellipse strain constraints + N buckling constraints + 1 frequency constraint
	dc = zeros(2*N, 1+3*N+2*nfreq);  % gradients of the constraints
else
	disp('problem type not implemented');
	error();
end
ceq = []; % must be specified!
dceq = []; % must be specified!

%% DISPLACEMENT
max_tip_disp = problem.constraints.max_tip_disp;
%Store value
c(1) = (u_tip-max_tip_disp) / scaling.char_disp;  % inequality constraint  c(x) <= 0
%Store gradient
for grd=1:problem.nvar
	dc( ((grd-1)*N+1):(grd*N) , 1 ) = dudx_tip.grad{grd} * scaling.char_var(grd) / scaling.char_disp;
end

%% STRAINS
%Store value
max_strain = problem.constraints.max_strain;
c(1, 2:N+1) = (strain(:,1) - max_strain(1)) ./ scaling.char_strain(1);
c(1, N+2:2*N+1) = (strain(:,2) - max_strain(2)) ./ scaling.char_strain(2);

%Store gradient
for grd=1:problem.nvar
	dc(((grd-1)*N+1):(grd*N),  2:N+1  ) = dstrain(1).grad{grd} * scaling.char_var(grd) / scaling.char_strain(1);
	dc(((grd-1)*N+1):(grd*N),N+2:2*N+1) = dstrain(2).grad{grd} * scaling.char_var(grd) / scaling.char_strain(2);
end

%% BUCKLING
eta_max = problem.constraints.max_buckling_factor;
%Store value
c(1, 2*N+2:3*N+1) = eta-eta_max;  % inequality constraints c(x) <= 0
%Store gradient
for grd=1:problem.nvar
	dc(((grd-1)*N+1):(grd*N),2*N+2:3*N+1) = deta.grad{grd} * scaling.char_var(grd);
end

%% FREQUENCIES

%Upper bound
for iw = 1:nfreq %Loop frequencies
    %Store value
    c(1,3*N+1+iw)=(w(iw)-(problem.constraints.max_freq(iw)*2*pi)^2) / scaling.char_freq;
    %Store gradient
	for grd=1:problem.nvar
		dc(((grd-1)*N+1):(grd*N),3*N+1+iw) = dw.grad{grd}(:,iw) * scaling.char_var(grd)/scaling.char_freq;
	end
end

%Lower bound
for iw = 1:nfreq %Loop frequencies
    %Store value
    c(1,3*N+1+nfreq+iw) = -(w(iw)-(problem.constraints.min_freq(iw)*2*pi)^2)/ scaling.char_freq;
    %Store gradient
	for grd=1:problem.nvar
		dc(((grd-1)*N+1):(grd*N),3*N+1+nfreq+iw) = -dw.grad{grd}(:,iw) * scaling.char_var(grd)/scaling.char_freq;
	end
end

% disp(dc)

end
