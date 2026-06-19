function [ m, dm ] = compute_mass( problem, design )

%% Cross section properties
[ csprops ] = compute_csprops( problem, design );

%% Compute mass constraint
%Calculate total mass
m = sum(csprops.total.me);	% m: total beam mass [kg]

%Calculate gradients
for grd=1:problem.nvar
	dm.grad{grd} = csprops.total.grad{grd}.me;
end

end
