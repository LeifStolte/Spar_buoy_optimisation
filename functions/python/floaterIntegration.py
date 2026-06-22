import numpy as np
np.trapz = np.trapezoid
from monopile import forceDistributed
from bisect import bisect_left as lookup
from floatingRotor import F_wind as F_wind_floating

def dqdt(t, q,
                structure,
                rotor,
                waves,
                wind):
    
    x1 = q[0:2]
    xdot1 = q[2:4]
    CT1 = q[4]
    rotor.CT = CT1
    
    # Extract time index
    i_ = lookup(waves.t, t)
    
    # Read wind speed
    V_hub = np.interp(t, wind.t, wind.V_hub)
    V_10 = wind.V_10
    
    # Nacelle speed
    x_dot_rotor = xdot1[0] + structure.zhub * xdot1[1]

    # Wind force if the rotor is on
    Thrust, CTVrel = F_wind_floating(rotor, V_10, V_hub, x_dot_rotor)
    Faero = np.array([Thrust, Thrust * structure.zhub])
    
    x_dot_submerged = xdot1[0] + structure.z * xdot1[1]
    
    u, ut = waves.u[i_,:], waves.ut[i_,:]
    df = forceDistributed(structure, u, ut, structure.z, x_dot_submerged)
    Fhydro = np.array([np.trapz(df, structure.z),
                       np.trapz(df * structure.z, structure.z)])
    
    output = np.zeros(5)
    output[0:2] = xdot1

    # Calculate the expression step by step
    # q = [x1; xdot1]
    # Left-hand side: (M + A) * x_dot2+ B * xdot1 + C * x1
    # Right-hand side: Fhydro + Faero
    # Solve for x_dot2 x_dot2 = inv(M + A) * (Fhydro + Faero - B * xdot1 - C * x1)

    B = structure.B
    C = structure.C
    output[2:4] = structure.MA_inv @ (Fhydro + Faero - B @ xdot1 - C @ x1)

    
    gamma = rotor.gamma
    output[4] = -gamma * (CT1 - CTVrel) 
    
    return output