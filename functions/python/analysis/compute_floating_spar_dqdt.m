function dqdt = compute_floating_spar_dqdt(t, q, structure, rotor, waves, wind)

x = q(1:2);
xdot = q(3:4);
ctState = q(5);

hubHeight = floating_spar_hub_height(structure, rotor);

V_hub = interp1(wind.t, wind.V_hub, t, 'linear', 'extrap');
u = interp1(waves.t, waves.u.', t, 'linear', 'extrap').';
ut = interp1(waves.t, waves.ut.', t, 'linear', 'extrap').';

xDotRotor = xdot(1) + hubHeight * xdot(2);
[thrust, ctRelative] = floating_spar_wind_force(rotor, V_hub, xDotRotor);
fAero = [thrust; thrust * hubHeight];

xDotSubmerged = xdot(1) + structure.z .* xdot(2);
fHydroDist = floating_spar_hydro_force(structure, u, ut, xDotSubmerged);
z = structure.z(:);
% ensure fHydroDist has z along rows
if size(fHydroDist,1) ~= numel(z)
    fHydroDist = reshape(fHydroDist, numel(z), []);
end

% manual trapezoidal integration to avoid trapz dimension issues
dz = diff(z);
if isempty(dz)
    % single point - integrals are zero
    f1 = 0.0;
    f2 = 0.0;
else
    y = fHydroDist;
    y1 = y(1:end-1, :);
    y2 = y(2:end, :);
    % integrate y over z
    f1_cols = sum((y1 + y2) .* (dz ./ 2), 1);
    % integrate y*z over z
    yz = y .* z;
    yz1 = yz(1:end-1, :);
    yz2 = yz(2:end, :);
    f2_cols = sum((yz1 + yz2) .* (dz ./ 2), 1);
    % reduce to scalars (if multiple columns, sum them)
    f1 = sum(f1_cols(:));
    f2 = sum(f2_cols(:));
end

fHydro = [f1; f2];

x = x(:);
xdot = xdot(:);
fHydro = fHydro(:);
fAero = fAero(:);

massMatrix = structure.M + structure.A;
inertiaForce = structure.B * xdot;
restoringForce = structure.C * x;
rhsForce = fHydro + fAero - inertiaForce - restoringForce;
accel = massMatrix \ rhsForce;

dqdt = zeros(5, 1);
dqdt(1:2) = xdot;
dqdt(3:4) = accel(:);
dqdt(5) = -rotor.gamma * (ctState - ctRelative);

end

function hubHeight = floating_spar_hub_height(structure, rotor)
if isfield(structure, 'zHub')
    hubHeight = structure.zHub;
elseif isfield(structure, 'z_Hub')
    hubHeight = structure.z_Hub;
elseif isfield(structure, 'zhub')
    hubHeight = structure.zhub;
elseif isfield(rotor, 'zHub')
    hubHeight = rotor.zHub;
else
    error('Floating spar hub height is missing. Expected zHub, z_Hub, or zhub in the structure or rotor data.');
end
end

function df = floating_spar_hydro_force(structure, u, ut, xDotSubmerged)
rhoWater = structure.rho_Water;
diameter = structure.DMonopile;

relativeVelocity = u - xDotSubmerged;
df = 0.5 * rhoWater * diameter * structure.CD .* abs(relativeVelocity) .* relativeVelocity ...
    + rhoWater * structure.CM * (pi / 4) * diameter^2 * ut;
end

function [thrust, ctRelative] = floating_spar_wind_force(rotor, V_hub, xDotRotor)
rhoAir = 1.22;
vRelative = V_hub - xDotRotor;
ctRelative = floating_spar_ct(rotor, vRelative);

if rotor.active
    thrust = floating_spar_favg(rotor, rhoAir) + floating_spar_fred(rotor) * ...
        (floating_spar_fvar(rotor, rhoAir, V_hub, xDotRotor, ctRelative) - floating_spar_favg(rotor, rhoAir));
else
    thrust = 0.0;
end
end

function ct = floating_spar_ct(rotor, v)
if v <= rotor.V1
    ct = rotor.Ct0;
elseif v <= rotor.VRated
    ct = rotor.Ct0 - rotor.c * (v - rotor.V1);
else
    ct = rotor.Ct1 * exp(-rotor.a * ((v - rotor.VRated)^rotor.b) / (10 + v - rotor.VRated)^rotor.b);
end
end

function value = floating_spar_fred(rotor)
if rotor.V_10 < rotor.VRated
    value = 0.54;
else
    value = 0.54 + 0.027 * (rotor.V_10 - rotor.VRated);
end
end

function value = floating_spar_favg(rotor, rhoAir)
value = 0.5 * rhoAir * rotor.ARotor * floating_spar_ct(rotor, rotor.V_10) * rotor.V_10^2;
end

function value = floating_spar_fvar(rotor, rhoAir, V_hub, xDotRotor, ct)
vRelative = V_hub - xDotRotor;
value = 0.5 * rhoAir * rotor.ARotor * ct * vRelative * abs(vRelative);
end