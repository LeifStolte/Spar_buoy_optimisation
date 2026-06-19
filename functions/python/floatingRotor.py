from dataclasses import dataclass

import numpy as np

from common import loadConstants
from rotor import Rotor, Ct, F_avg, fRed


c = loadConstants()


@dataclass
class FloatingRotor(Rotor):
    def F_wind(self, V_10, V_hub, x_dot=0.0):
        V_rel = V_hub - x_dot
        CTVrel = Ct(self, V_rel)

        if self.active:
            F_wind_out = F_avg(self, V_10) + fRed(self, V_10) * (F_var(self, V_hub, x_dot, CTVrel) - F_avg(self, V_10))
            return F_wind_out, CTVrel
        return 0.0, CTVrel

    def F_var(self, V_hub, x_dot=0.0, CT=0.0):
        V_rel = V_hub - x_dot
        return 0.5 * c.rho_air * self.ARotor * CT * V_rel * np.abs(V_rel)


def F_wind(rotorDict, V_10, V_hub, x_dot=0.0):
    if isinstance(rotorDict, FloatingRotor):
        rotor = rotorDict
    elif isinstance(rotorDict, Rotor):
        rotor = FloatingRotor.from_mapping(rotorDict.to_mapping())
    else:
        rotor = FloatingRotor.from_mapping(rotorDict)
    return rotor.F_wind(V_10, V_hub, x_dot)


def F_var(rotorDict, V_hub, x_dot=0.0, CT=0.0):
    if isinstance(rotorDict, FloatingRotor):
        rotor = rotorDict
    elif isinstance(rotorDict, Rotor):
        rotor = FloatingRotor.from_mapping(rotorDict.to_mapping())
    else:
        rotor = FloatingRotor.from_mapping(rotorDict)
    return rotor.F_var(V_hub, x_dot, CT)

