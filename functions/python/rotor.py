from dataclasses import dataclass

import numpy as np

from common import loadConstants
from models import Model


c = loadConstants()


@dataclass
class Rotor(Model):
    V1: float = 0.0
    Ct0: float = 0.0
    c: float = 0.0
    VRated: float = 0.0
    Ct1: float = 0.0
    a: float = 0.0
    b: float = 0.0
    ARotor: float = 0.0
    gamma: float = 0.0
    CT: float = 0.0
    active: bool = True

    def Ct(self, V):
        if V <= self.V1:
            return self.Ct0
        if np.logical_and(V > self.V1, V <= self.VRated):
            return self.Ct0 - self.c * (V - self.V1)
        return self.Ct1 * np.exp(-self.a * ((V - self.VRated) ** self.b) / (10 + V - self.VRated) ** self.b)

    def fRed(self, V_10):
        if V_10 < self.VRated:
            return 0.54
        return 0.54 + 0.027 * (V_10 - self.VRated)

    def F_avg(self, V_10):
        return 0.5 * c.rho_air * self.ARotor * self.Ct(V_10) * V_10 ** 2

    def F_var(self, V_hub, x_dot=0.0):
        V_rel = V_hub - x_dot
        return 0.5 * c.rho_air * self.ARotor * self.Ct(V_hub) * V_rel * np.abs(V_rel)

    def F_wind(self, V_10, V_hub, x_dot):
        return self.F_avg(V_10) + self.fRed(V_10) * (self.F_var(V_hub, x_dot) - self.F_avg(V_10))


def Ct(rotorDict, V):
    rotor = rotorDict if isinstance(rotorDict, Rotor) else Rotor.from_mapping(rotorDict)
    return rotor.Ct(V)


def fRed(rotorDict, V_10):
    rotor = rotorDict if isinstance(rotorDict, Rotor) else Rotor.from_mapping(rotorDict)
    return rotor.fRed(V_10)


def F_avg(rotorDict, V_10):
    rotor = rotorDict if isinstance(rotorDict, Rotor) else Rotor.from_mapping(rotorDict)
    return rotor.F_avg(V_10)


def F_var(rotorDict, V_hub, x_dot=0.0):
    rotor = rotorDict if isinstance(rotorDict, Rotor) else Rotor.from_mapping(rotorDict)
    return rotor.F_var(V_hub, x_dot)


def F_wind(rotorDict, V_10, V_hub, x_dot):
    rotor = rotorDict if isinstance(rotorDict, Rotor) else Rotor.from_mapping(rotorDict)
    return rotor.F_wind(V_10, V_hub, x_dot)