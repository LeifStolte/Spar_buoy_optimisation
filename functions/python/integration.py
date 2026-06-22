import numpy as np
np.trapz = np.trapezoid
from bisect import bisect_left as lookup

from floatingRotor import F_wind as F_wind_floating
from monopile import forceDistributed


class Integrator:
    def rk4(self, f, h, y0, t0, *args, **kwargs):
        k1 = f(t0, y0, *args, **kwargs)
        k2 = f(t0 + h / 2, y0 + h / 2 * k1, *args, **kwargs)
        k3 = f(t0 + h / 2, y0 + h / 2 * k2, *args, **kwargs)
        k4 = f(t0 + h, y0 + h * k3, *args, **kwargs)
        return y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def ode4(self, odefun, tspan, y0, *args, **kwargs):
        y = np.zeros((len(tspan), len(y0)))
        y[0, :] = y0
        h = np.diff(tspan)
        y_ini = y0

        for i_, t_ in enumerate(tspan[:-1]):
            hi = h[i_]
            y_integrated = self.rk4(odefun, hi, y_ini, t_, *args, **kwargs)
            y_ini = y_integrated
            y[i_ + 1, :] = y_integrated

        return y

    def dqdt(self, t, q, structure, rotor, waves, wind, arg):
        alpha, alphaDot = q
        GF = self.GFCalc(t, alphaDot, structure, rotor, waves, wind, arg)

        M = structure.GM
        D = structure.GD
        K = structure.GK

        alphaDotDot = (1.0 / M) * (-D * alphaDot - K * alpha + GF)

        dqdtOut = np.zeros(2)
        dqdtOut[0] = alphaDot
        dqdtOut[1] = alphaDotDot
        return dqdtOut

    def GFCalc(self, t, alphaDot, structure, rotor, waves, wind, arg):
        x_dot = alphaDot * structure.phiNodal

        wetPart = np.less_equal(structure.zBeamNodal, 0.0)
        x_dot_submerged = x_dot[wetPart]
        phiNodalSubmerged = structure.phiNodal[wetPart]

        i_ = lookup(waves.t, t)
        u, ut, z = waves.u[i_, :], waves.ut[i_, :], waves.z
        df = forceDistributed(structure, u, ut, z, x_dot_submerged)
        GFWaves = np.trapz(df * phiNodalSubmerged, z)

        if arg == "onlyWaves":
            return GFWaves

        i_ = lookup(wind.t, t)
        V_10, V_hub = wind.V_10, wind.V_hub[i_]
        phiHub = structure.phiNodal[-1]
        x_dot_rotor = alphaDot * phiHub
        F_aero, _ = F_wind_floating(rotor, V_10, V_hub, x_dot_rotor)
        GFWind = F_aero * phiHub
        return float(GFWind + GFWaves)


def rk4(f, h, y0, t0, *args, **kwargs):
    return Integrator().rk4(f, h, y0, t0, *args, **kwargs)


def ode4(odefun, tspan, y0, *args, **kwargs):
    return Integrator().ode4(odefun, tspan, y0, *args, **kwargs)


def dqdt(t, q, structure, rotor, waves, wind, arg):
    return Integrator().dqdt(t, q, structure, rotor, waves, wind, arg)


def GFCalc(t, alphaDot, structure, rotor, waves, wind, arg):
    return Integrator().GFCalc(t, alphaDot, structure, rotor, waves, wind, arg)