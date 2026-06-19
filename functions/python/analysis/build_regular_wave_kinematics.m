function waves = build_regular_wave_kinematics(waveInput, timeInfo, z)

g = 9.81;

t = (0:timeInfo.dt:(timeInfo.TDur - timeInfo.dt)).';
z = z(:).';

amplitude = 0.5 * waveInput.Hs;
omega = 2 * pi / waveInput.Tp;
k = omega^2 / g;
transfer = cosh(k * (z + waveInput.h)) ./ sinh(k * waveInput.h);

eta = amplitude * cos(omega * t);
u = -amplitude * omega * cos(omega * t) * transfer;
ut = amplitude * omega^2 * sin(omega * t) * transfer;

waves.t = t;
waves.z = z;
waves.eta = eta;
waves.u = u;
waves.ut = ut;

end