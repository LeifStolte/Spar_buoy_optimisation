function [ csprops ] = compute_caps_csprops( problem, design )

%% Parameters
utils.E = problem.materialdata(1).E1;
utils.G = problem.materialdata(1).G12;
utils.rho = problem.materialdata(1).rho;
utils.l = problem.info.delta_z;
utils.h = problem.geometry.h;
utils.c = problem.geometry.c;
utils.N = problem.info.num_sec;

%% Values
%area
csprops.A = 2*design.t.*design.a ;	% beam area [m^2]
Ix = design.a .* design.t.^3/6 + design.a.*design.t/2.*(utils.h-design.t).^2;
Iy = 2 * design.t .* design.a.^3/12;	% second moments of area [m^4]
csprops.K = zeros(size(design.a,1),1);
%stiffness
csprops.EA = utils.E * csprops.A;
csprops.kGA = utils.h * 0.05 * utils.G; %Adding some artifically high shear stiffness to avoid zero shear stiffness
csprops.EIx = utils.E*Ix;
csprops.EIy = utils.E*Iy;	% second moments of area [m^4]
csprops.GK = zeros(size(design.t,1),1);	% torsion constant [m^4]
%mass
csprops.ms = csprops.A .* utils.rho; % mass per unit length
csprops.me = utils.l * csprops.ms; %element mass
csprops.Ix = utils.rho.*Ix;
csprops.Iy = utils.rho.*Iy;

%% Gradients
%area
csprops.grad{2}.A = 2 * design.a;
csprops.grad{1}.A = 2 * design.t;
csprops.grad{3}.A = zeros(size(design.a,1),1);

dIx = capsdIx( utils, design );
grad{2}.Ix = dIx(:,1); %design.a .* design.t.^2/2 + design.a/2.*(utils.h-design.t).^2-design.a.*design.t.*(utils.h-design.t); %dIx(:,1); 
grad{1}.Ix = dIx(:,2); %design.t.^3/6 + design.t/2.*(utils.h-design.t).^2; %dIx(:,2); 
grad{3}.Ix = zeros(size(design.a,1),1);

dIy = capsdIy( design );	% second moments of area [m^4]
grad{2}.Iy = dIy(:,1);
grad{1}.Iy = dIy(:,2);
grad{3}.Iy = zeros(size(design.a,1),1);

csprops.grad{2}.K = zeros(size(design.a,1),1);
csprops.grad{1}.K = zeros(size(design.a,1),1);
csprops.grad{3}.K = zeros(size(design.a,1),1);

%stiffness
csprops.grad{2}.EA = utils.E * csprops.grad{2}.A;
csprops.grad{1}.EA = utils.E * csprops.grad{1}.A;
csprops.grad{3}.EA = zeros(size(design.e,1),1);

csprops.grad{2}.kGA = zeros(size(design.a,1),1);
csprops.grad{1}.kGA = zeros(size(design.t,1),1);
csprops.grad{3}.kGA = zeros(size(design.e,1),1);

csprops.grad{2}.EIx = utils.E*grad{2}.Ix;
csprops.grad{1}.EIx = utils.E*grad{1}.Ix;
csprops.grad{3}.EIx = zeros(size(design.e,1),1);

csprops.grad{2}.EIy = utils.E*grad{2}.Iy;
csprops.grad{1}.EIy = utils.E*grad{1}.Iy;
csprops.grad{3}.EIy = zeros(size(design.e,1),1);

csprops.grad{2}.GK = zeros(size(design.t,1),1);	% torsion constant [m^4]
csprops.grad{1}.GK = zeros(size(design.a,1),1);	% torsion constant [m^4]
csprops.grad{3}.GK = zeros(size(design.e,1),1);	% torsion constant [m^4]

%mass
csprops.grad{2}.ms = csprops.grad{2}.A .* utils.rho; % mass per unit length
csprops.grad{1}.ms = csprops.grad{1}.A .* utils.rho; % mass per unit length
csprops.grad{3}.ms = zeros(size(design.e,1),1); % mass per unit length

csprops.grad{2}.me = utils.l * csprops.grad{2}.ms; %element mass
csprops.grad{1}.me = utils.l * csprops.grad{1}.ms; %element mass
csprops.grad{3}.me = zeros(size(design.a,1),1);%element mass

csprops.grad{2}.Ix = utils.rho.*grad{2}.Ix;
csprops.grad{1}.Ix = utils.rho.*grad{1}.Ix;
csprops.grad{3}.Ix = utils.rho.*grad{3}.Ix;

csprops.grad{2}.Iy = utils.rho.*grad{2}.Iy;
csprops.grad{1}.Iy = utils.rho.*grad{1}.Iy;
csprops.grad{3}.Iy = utils.rho.*grad{3}.Iy;

    function [ dIx ] = capsdIx( utils, design)
        
        dIx = zeros(size(design.t,1),2);
        
        for s=1:size(design.t,1)
            t = design.t(s);
            a = design.a(s);
            h = utils.h(s);
            
            t1 = t ^ 2;
            t4 = h - t;
            t5 = t4 ^ 2;
            dIx(s,1) = a * t1 / 0.2e1 + a * t5 / 0.2e1 - a * t * t4;
            dIx(s,2) = t1 * t / 0.6e1 + t * t5 / 0.2e1;
            
        end
        
    end

    function [ dIy ] = capsdIy( design )
        
        dIy = zeros(size(design.t,1),2);
        
        for s=1:size(design.t,1)
            t = design.t(s);
            a = design.a(s);
            
            t1 = a ^ 2;
            dIy(s,1) = a * t1 / 0.6e1;
            dIy(s,2) = t1 * t / 0.2e1;
        end
        
        
    end

end

