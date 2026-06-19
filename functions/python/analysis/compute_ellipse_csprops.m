function [ csprops ] = compute_ellipse_csprops( problem, design )

csprops.grad = cell(problem.nvar,1);
for i=1:problem.nvar
	csprops.grad{i}=struct('A', [], 'K', [], 'EA', [], 'kGA', [], 'EIx', [], 'EIy', [], 'GK', [], 'ms', [], 'me', [], 'Ix', [], 'Iy', []);
end

%% Parameters
utils.E = problem.materialdata(2).E1;
utils.G = problem.materialdata(2).G12;
utils.rho = problem.materialdata(2).rho;
utils.l = problem.info.delta_z;
utils.h = problem.geometry.h;
utils.c = problem.geometry.c;
utils.N = problem.info.num_sec;

k=0.53;  %shear correction factor

%% Values
%area
csprops.A = ellipseA( utils, design);	% beam area [m^2]
Ix = ellipseIx( utils, design);
Iy = ellipseIy( utils, design);	% second moments of area [m^4]
csprops.K = ellipseK( utils, design);
%stiffness
csprops.EA = utils.E * csprops.A;
csprops.kGA = k * utils.G * csprops.A;
csprops.EIx = utils.E*Ix;
csprops.EIy = utils.E*Iy;	% second moments of area [m^4]
csprops.GK = utils.G*csprops.K;	% torsion constant [m^4]
%mass
csprops.ms = csprops.A .* utils.rho; % mass per unit length
csprops.me = utils.l * csprops.ms; %element mass
csprops.Ix = utils.rho.*Ix;
csprops.Iy = utils.rho.*Iy;

%% Gradients

%area
dA = ellipsedA( utils, design);
csprops.grad{2}.A = zeros(size(design.e,1),1);
csprops.grad{1}.A = zeros(size(design.e,1),1);
csprops.grad{3}.A = dA(:,1);

dIx = ellipsedIx( utils, design);
grad{2}.Ix = zeros(size(design.e,1),1);
grad{1}.Ix = zeros(size(design.e,1),1);
grad{3}.Ix = dIx(:,1);

dIy = ellipsedIy( utils, design);	% second moments of area [m^4]
grad{2}.Iy = zeros(size(design.e,1),1);
grad{1}.Iy = zeros(size(design.e,1),1);
grad{3}.Iy = dIy(:,1);

dK = ellipsedK( utils, design);	% torsion constant [m^4]
csprops.grad{2}.K = zeros(size(design.e,1),1);
csprops.grad{1}.K = zeros(size(design.e,1),1);
csprops.grad{3}.K = dK(:,1);

%stiffness
csprops.grad{2}.EA = zeros(size(design.e,1),1);
csprops.grad{1}.EA = zeros(size(design.e,1),1);
csprops.grad{3}.EA = utils.E * csprops.grad{3}.A;

csprops.grad{2}.kGA = zeros(size(design.e,1),1);
csprops.grad{1}.kGA = zeros(size(design.e,1),1);
csprops.grad{3}.kGA = k * utils.G * csprops.grad{3}.A;

csprops.grad{2}.EIx = zeros(size(design.e,1),1);
csprops.grad{1}.EIx = zeros(size(design.e,1),1);
csprops.grad{3}.EIx = utils.E*grad{3}.Ix;

csprops.grad{2}.EIy = zeros(size(design.e,1),1);
csprops.grad{1}.EIy = zeros(size(design.e,1),1);
csprops.grad{3}.EIy = utils.E*grad{3}.Iy;

csprops.grad{2}.GK = zeros(size(design.e,1),1);	% torsion constant [m^4]
csprops.grad{1}.GK = zeros(size(design.a,1),1);	% torsion constant [m^4]
csprops.grad{3}.GK = utils.G*csprops.grad{3}.K;

%mass
csprops.grad{2}.ms = zeros(size(design.e,1),1); % mass per unit length
csprops.grad{1}.ms = zeros(size(design.e,1),1); % mass per unit length
csprops.grad{3}.ms = csprops.grad{3}.A .* utils.rho; % mass per unit length

csprops.grad{2}.me = zeros(size(design.e,1),1); %element mass
csprops.grad{1}.me = zeros(size(design.e,1),1); %element mass
csprops.grad{3}.me = utils.l * csprops.grad{3}.ms; %element mass

csprops.grad{2}.Ix = utils.rho.*grad{2}.Ix;
csprops.grad{1}.Ix = utils.rho.*grad{1}.Ix;
csprops.grad{3}.Ix = utils.rho.*grad{3}.Ix;

csprops.grad{2}.Iy = utils.rho.*grad{2}.Iy;
csprops.grad{1}.Iy = utils.rho.*grad{1}.Iy;
csprops.grad{3}.Iy = utils.rho.*grad{3}.Iy;

%% Values
    function [ A ] = ellipseA( utils, design)
        
        A = zeros(size(design.e,1),1);
        
        for s=1:size(design.e,1)
            t = design.e(s);
            a = utils.h(s)/2;
            b = utils.c(s)/2;
            
            A(s) = pi * t * (a + b) * (0.1e1 + (0.2464e0 + 0.2222e-2 * a / b + 0.2222e-2 * b / a) * (a - b) ^ 2 / (a + b) ^ 2);
        end
    end

    function [ Ix ] = ellipseIx( utils, design)
        
        Ix = zeros(size(design.e,1),1);
        
        for s=1:size(design.e,1)
            t = design.e(s);
            a = utils.h(s)/2;
            b = utils.c(s)/2;
            
            t2 = a ^ 2;
            t9 = b ^ 2;
            t15 = (a - b) ^ 2;
            t18 = (a + b) ^ 2;
            t19 = 0.1e1 / t18;
            t25 = t ^ 2;
            Ix(s) = pi * t * t2 * (a + 3 * b) * (0.1e1 + (0.1349e0 + 0.1279e0 * a / b - 0.1284e-1 * t2 / t9) * t15 * t19) / 0.4e1 + pi * t25 * t * (3 * a + b) * (0.1e1 + (0.1349e0 + 0.1279e0 * b / a - 0.1284e-1 * t9 / t2) * t15 * t19) / 0.16e2;
            
        end
        
    end

    function [ Iy ] = ellipseIy( utils, design)
        
        Iy = zeros(size(design.e,1),1);
        
        for s=1:size(design.e,1)
            t = design.e(s);
            a = utils.h(s)/2;
            b = utils.c(s)/2;
            
            t2 = b ^ 2;
            t9 = a ^ 2;
            t15 = (b - a) ^ 2;
            t18 = (a + b) ^ 2;
            t19 = 0.1e1 / t18;
            t25 = t ^ 2;
            Iy(s) = pi * t * t2 * (3 * a + b) * (0.1e1 + (0.1349e0 + 0.1279e0 * b / a - 0.1284e-1 * t2 / t9) * t15 * t19) / 0.4e1 + pi * t25 * t * (a + 3 * b) * (0.1e1 + (0.1349e0 + 0.1279e0 * a / b - 0.1284e-1 * t9 / t2) * t15 * t19) / 0.16e2;
            
        end
    end

    function [ K ] = ellipseK( utils, design)
        
        K = zeros(size(design.e,1),1);
        
        for s=1:size(design.e,1)
            t = design.e(s);
            a = utils.h(s)/2;
            b = utils.c(s)/2;
            
            t1 = pi ^ 2;
            t3 = 0.5e0 * t;
            t5 = (a - t3) ^ 2;
            t8 = (b - t3) ^ 2;
            t11 = a + b - t;
            t14 = (a - b) ^ 2;
            t15 = t11 ^ 2;
            K(s) = 0.4e1 * t1 * t * t5 * t8 / pi / t11 / (0.1e1 + 0.258e0 * t14 / t15);
        end
    end

%% Gradients
    function [ dA ] = ellipsedA( utils, design)
        
        dA = zeros(size(design.e,1),1);
        
        for s=1:size(design.e,1)
            t = design.e(s);
            a = utils.h(s)/2;
            b = utils.c(s)/2;
            
            t1 = a + b;
            t11 = (a - b) ^ 2;
            t13 = t1 ^ 2;
            dA(s) = pi * t1 * (0.1e1 + (0.2464e0 + 0.2222e-2 * a / b + 0.2222e-2 * b / a) * t11 / t13);
            
        end
    end

    function [ dIx ] = ellipsedIx( utils, design)
        
        dIx = zeros(size(design.e,1),1);
        
        for s=1:size(design.e,1)
            t = design.e(s);
            a = utils.h(s)/2;
            b = utils.c(s)/2;
            
            t1 = a ^ 2;
            t8 = b ^ 2;
            t14 = (a - b) ^ 2;
            t17 = (a + b) ^ 2;
            t18 = 0.1e1 / t17;
            t24 = t ^ 2;
            dIx(s) = pi * t1 * (a + 3 * b) * (0.1e1 + (0.1349e0 + 0.1279e0 * a / b - 0.1284e-1 * t1 / t8) * t14 * t18) / 0.4e1 + 0.3e1 / 0.16e2 * pi * t24 * (3 * a + b) * (0.1e1 + (0.1349e0 + 0.1279e0 * b / a - 0.1284e-1 * t8 / t1) * t14 * t18);
            
        end
        
    end

    function [ dIy ] = ellipsedIy( utils, design)
        
        dIy = zeros(size(design.e,1),1);
        
        for s=1:size(design.e,1)
            t = design.e(s);
            a = utils.h(s)/2;
            b = utils.c(s)/2;
            
            t1 = b ^ 2;
            t8 = a ^ 2;
            t14 = (b - a) ^ 2;
            t17 = (a + b) ^ 2;
            t18 = 0.1e1 / t17;
            t24 = t ^ 2;
            dIy(s) = pi * t1 * (3 * a + b) * (0.1e1 + (0.1349e0 + 0.1279e0 * b / a - 0.1284e-1 * t1 / t8) * t14 * t18) / 0.4e1 + 0.3e1 / 0.16e2 * pi * t24 * (a + 3 * b) * (0.1e1 + (0.1349e0 + 0.1279e0 * a / b - 0.1284e-1 * t8 / t1) * t14 * t18);
            
        end
        
    end

    function [ dK ] = ellipsedK( utils, design)
        
        dK = zeros(size(design.e,1),1);
        
        for s=1:size(design.e,1)
            t = design.e(s);
            a = utils.h(s)/2;
            b = utils.c(s)/2;
            
            t1 = pi ^ 2;
            t2 = 0.5e0 * t;
            t3 = a - t2;
            t4 = t3 ^ 2;
            t6 = b - t2;
            t7 = t6 ^ 2;
            t9 = 0.1e1 / pi;
            t10 = a + b - t;
            t11 = 0.1e1 / t10;
            t14 = (a - b) ^ 2;
            t15 = t10 ^ 2;
            t16 = 0.1e1 / t15;
            t19 = 0.1e1 + 0.258e0 * t14 * t16;
            t20 = 0.1e1 / t19;
            t24 = t1 * t;
            t26 = t7 * t9;
            t27 = t11 * t20;
            t31 = t24 * t4;
            t42 = t15 ^ 2;
            t45 = t19 ^ 2;
            dK(s) = 0.4e1 * t1 * t4 * t7 * t9 * t11 * t20 - 0.40e1 * t24 * t3 * t26 * t27 - 0.40e1 * t31 * t6 * t9 * t27 + 0.4e1 * t31 * t26 * t16 * t20 - 0.2064e1 * t24 * t4 * t7 * t9 / t42 / t45 * t14;
            
        end
        
    end

end

