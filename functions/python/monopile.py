from dataclasses import dataclass

import numpy as np

from common import loadConstants
from models import Model


g = loadConstants().g
rho_water = loadConstants().rho_water


@dataclass
class Monopile(Model):
    DMonopile: float = 0.0
    CD: float = 0.0
    CM: float = 0.0
    zBeamNodal: any = None
    phiNodal: any = None
    rhoAElement: any = None
    zBeamElement: any = None
    dz: any = None
    phiElement: any = None

    def force_integrate(self, u, ut, z, x_dot):
        h = np.abs(z[0])
        u = u - x_dot
        df = self.force_distributed(u, ut, z, x_dot)
        F = np.trapz(df, z)
        M = np.trapz(df * (h + z), z)
        return F, M

    def force_distributed(self, u, ut, z, x_dot):
        del z
        u = u - x_dot
        df_drag = 0.5 * rho_water * self.DMonopile * self.CD * np.abs(u) * u
        df_inertia = rho_water * self.CM * (np.pi / 4) * self.DMonopile ** 2 * ut
        return df_drag + df_inertia

    def compute_elementwise_quantities(self):
        z = self.zBeamNodal
        dz = np.diff(z)
        self.zBeamElement = z[:-1] + dz / 2
        self.dz = dz

        phi = self.phiNodal
        dPhi = np.diff(phi)
        self.phiElement = self.phiNodal[:-1] + dPhi / 2
        return self


def forceIntegrate(monopileDict, u, ut, z, x_dot):
    monopile = monopileDict if isinstance(monopileDict, Monopile) else Monopile.from_mapping(monopileDict.to_mapping() if hasattr(monopileDict, "to_mapping") else monopileDict)
    return monopile.force_integrate(u, ut, z, x_dot)


def forceDistributed(monopileDict, u, ut, z, x_dot):
    monopile = monopileDict if isinstance(monopileDict, Monopile) else Monopile.from_mapping(monopileDict.to_mapping() if hasattr(monopileDict, "to_mapping") else monopileDict)
    return monopile.force_distributed(u, ut, z, x_dot)


def computeElementwiseQuantities(monopileDict):
    monopile = monopileDict if isinstance(monopileDict, Monopile) else Monopile.from_mapping(monopileDict.to_mapping() if hasattr(monopileDict, "to_mapping") else monopileDict)
    return monopile.compute_elementwise_quantities()
    