function [ strain, dstrain ] = compute_strains( problem, design, csprops, solution )


% This controls whether the gradient will take the gradient wrt the geometry
grad_wrt_geom = 0;

% Take the geometry from the appropriate data structure according to the problem
if strcmp(problem.type,'blade_structure')

	H = problem.geometry.h;
	W = problem.geometry.c;
	%Aux
	EIx = csprops.total.EIx; %Bending stiffness around x axis
	EIy = csprops.total.EIy; %Bending stiffness around y axis

elseif strcmp(problem.type, 'tower_monopile')

	H = design.D;
	W = design.D;
	grad_wrt_geom=1;
	%Aux
	EIx = csprops.total.areaIx; %Bending stiffness around x axis
	EIy = csprops.total.areaIy; %Bending stiffness around y axis

else

	disp('problem type not implemented');
	error();

end


%Value
%Extract cross section moments
Mx = solution.f(4,:,1)'; %based on edgewise load case
My = solution.f(5,:,2)'; %based on flapwise load case

% Get the sign matrix
N=size(Mx,1);
signMx=ones(N,1);
signMy=ones(N,1);
for i=1:N
	if Mx(i)<0.0
		signMx(i)=-1.0;
	end
	if My(i)<0.0
		signMy(i)=-1.0;
	end
end

% Compute section curvature
kappax = abs(Mx)./(EIx);
kappay = abs(My)./(EIy);

% Compute bending strains
strain(:,1) = kappax.*H/2; %caps
strain(:,2) = kappay.*W/2; %ellipse

%M*y/I
%
%M'*s*y/I + M*y'/I

%Gradient
% Construct the gradients of the curvature
%Note tha E is also squared and so we multiply by EIxda

for grd=1:problem.nvar

	dMx=zeros(N);
	dMy=zeros(N);
	dMx(:,:)=solution.grad{grd}.f(4,:,1,:);
	dMy(:,:)=solution.grad{grd}.f(5,:,2,:);
	%dMx
	%dMy

	% Take the geometry from the appropriate data structure according to the problem
	if strcmp(problem.type,'blade_structure')
		dEIx = csprops.total.grad{grd}.EIx; %Bending stiffness around x axis
		dEIy = csprops.total.grad{grd}.EIy; %Bending stiffness around y axis
	elseif strcmp(problem.type, 'tower_monopile')
		dEIx = csprops.total.grad{grd}.areaIx; %Bending stiffness around x axis
		dEIy = csprops.total.grad{grd}.areaIy; %Bending stiffness around y axis
	else
		disp('problem type not implemented');
		error();
	end

	dkappax.grad{grd} = (dMx')*diag(signMx./(EIx)) + diag( -abs(Mx) ./ (EIx.^2) .* ( dEIx ) );
	dkappay.grad{grd} = (dMy')*diag(signMy./(EIy)) + diag( -abs(My) ./ (EIy.^2) .* ( dEIy ) );

	% Compute the gradients of the bending strain
	dstrain(1).grad{grd} = dkappax.grad{grd} * diag(H/2); %caps
	dstrain(2).grad{grd} = dkappay.grad{grd} * diag(W/2); %ellipse

	% If the DV effects geometry, we need to add an extra term
	if grad_wrt_geom==1 & grd==1
		dstrain(1).grad{grd} = dstrain(1).grad{grd} + 0.5*diag(kappax); %caps
		dstrain(2).grad{grd} = dstrain(2).grad{grd} + 0.5*diag(kappay); %ellipse
	end

end

%global fd_data;
%if fd_data.state==1
%	%fd_data.data= [ dstrain(1).grad{1} dstrain(2).grad{1} ; dstrain(1).grad{2} dstrain(2).grad{2} ];
%	fd_data.data= [ dkappax.grad{1} dkappay.grad{1} ; dkappax.grad{2} dkappay.grad{2} ];
%	%fd_data.data
%	%fd_data.data=zeros(40,size(Mx,1));
%elseif fd_data.state==2
%	%fd_data.data=[ strain(:,1) ; strain(:,2) ]';
%	fd_data.data=[ kappax ; kappay ]';
%	%fd_data.data=Mx';
%	%fd_data.data
%elseif fd_data.state==3
%	%fd_data.data=[ strain(:,1) ; strain(:,2) ]';
%	fd_data.data=[ kappax ; kappay ]';
%	%fd_data.data=Mx';
%	%fd_data.data
%end

end
