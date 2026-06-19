function [ csprops ] = compute_solidCsprops( utils, design )
%COMPUTE_SOLIDCSPROPS Summary of this function goes here
%   Detailed explanation goes here

%% Values
%area
csprops.A = design.t.*utils.h ;	% beam area [m^2]
csprops.Ix = 1/12 * utils.h.^3 .* design.t ;	
csprops.Iy = 1/12 * design.t.^3 .* utils.h;	% second moments of area [m^4]
csprops.ASx = 5/6*csprops.A ;	% effective area of shear
csprops.ASy = 5/6*csprops.A ;	% effective area of shear
a=0.5*max([utils.h design.t],[],2); b=0.5*min([utils.h design.t],[],2); 
csprops.J =  a.*b.^3 .* (16/3-3.36*b./a .* (1-b.^4 ./ (12*a.^4))); % torsion constant [m^4]
%stiffness
csprops.EA = utils.E * csprops.A; 
csprops.kGA = utils.k * utils.G * csprops.A; 
csprops.EIx = utils.E*csprops.Ix;
csprops.EIy = utils.E*csprops.Iy;	% second moments of area [m^4]
csprops.GJ = utils.G*csprops.J;	% torsion constant [m^4]
%mass
csprops.ms = csprops.A * utils.rho; %element mass
csprops.me = utils.l * csprops.ms; %element mass

%% Gradients

% %% Gradients
% %area
% csprops.dA.dt = 2 * design.a;
% csprops.dA.da = 2 * design.t;
% dIx = capsdIx( utils, design);
% csprops.dIx.dt = dIx(:,1);
% csprops.dIx.da = dIx(:,2);
% dIy = capsdIy( utils, design);	% second moments of area [m^4]
% csprops.dIy.dt = dIy(:,1);
% csprops.dIy.da = dIy(:,2);
% %stiffness
% csprops.dEA = utils.E * csprops.A; 
% csprops.dkGA.dt = utils.k * utils.G * csprops.dA.dt; 
% csprops.dkGA.da = utils.k * utils.G * csprops.dA.da; 
% csprops.dEIx.dt = utils.E*csprops.dIx.dt;
% csprops.dEIx.da = utils.E*csprops.dIx.da;
% csprops.dEIy.dt = utils.E*csprops.dIy.dt;
% csprops.dEIy.da = utils.E*csprops.dIy.da;
% csprops.dGJ.dt = zeros(size(design.t,1),1);	% torsion constant [m^4]
% csprops.dGJ.da = zeros(size(design.a,1),1);	% torsion constant [m^4]
% %mass
% csprops.dms.dt = csprops.dA.dt .* utils.rho; % mass per unit length
% csprops.dms.da = csprops.dA.da .* utils.rho; % mass per unit length
% csprops.dme.dt = utils.l * csprops.dms.dt; %element mass
% csprops.dme.da = utils.l * csprops.dms.da; %element mass

end


