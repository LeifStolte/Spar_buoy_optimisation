import numpy as np

from monopile import forceDistributed
from models import LoadSeries


def calculateStaticWindLoads(wind, rotor, structure, q):
    del rotor, q

    F_with_rel_motion = np.zeros_like(wind.t)
    for i_, _ in enumerate(wind.t):
        F_with_rel_motion[i_] = 0.0

    z = structure.zBeamNodal
    h = np.abs(z[0])
    zHub = z[-1]
    return LoadSeries(t=wind.t, F=F_with_rel_motion, M=F_with_rel_motion * (zHub + h))


def calculateStaticWaveLoads(waves, structure, q):
    x_dot = q.alphaDot[:, None] * structure.phiNodal[None, :]

    wetPart = np.less_equal(structure.zBeamNodal, 0.0)
    x_dot_submerged = x_dot[:, wetPart]
    zSubmerged = structure.zBeamNodal[wetPart]
    h = np.abs(zSubmerged[0])

    F_with_rel_motion = np.zeros_like(waves.t)
    M_with_rel_motion = np.zeros_like(waves.t)

    for i_, _ in enumerate(waves.t):
        u, ut, z = waves.u[i_, :], waves.ut[i_, :], waves.z
        df = forceDistributed(structure, u, ut, z, x_dot_submerged[i_, :])
        F_with_rel_motion[i_] = np.trapz(df, zSubmerged)
        M_with_rel_motion[i_] = np.trapz(df * (zSubmerged + h), zSubmerged)

    return LoadSeries(t=waves.t, F=F_with_rel_motion, M=M_with_rel_motion)


def calculateDynamicLoads(structure, q):
    F = np.zeros_like(q.t)
    M = np.zeros_like(q.t)

    dz = structure.dz
    zElement = structure.zBeamElement
    h = np.abs(structure.zBeamNodal[0])
    phiElement = structure.phiElement
    rhoA = structure.rhoAElement

    for i_, _ in enumerate(q.t):
        M[i_] = np.sum(-rhoA * dz * (zElement + h) * phiElement * q.alphaDotDot[i_])
        F[i_] = np.sum(-rhoA * dz * phiElement * q.alphaDotDot[i_])

    return LoadSeries(t=q.t, F=F, M=M)