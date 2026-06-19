from rotor import F_avg, fRed, Ct
from common import loadConstants
import numpy as np

c = loadConstants()

def F_wind(rotorDict, V_10, V_hub, x_dot=0.):
    
    # FIXME Assignment 5 Question 2.13: calculate V_rel
    V_rel = V_hub-x_dot
    
    # FIXME Assignment 5 Question 2.13 make the call to CT function dependent on V_rel
    CTVrel = Ct(rotorDict, V_rel)
    
    # If the controller is active (gamma!=0), then CT comes from the rotor dict.
    # Otherwise, compute it on V_rel
    if np.isclose(rotorDict["gamma"], 0.):
        CT = CTVrel
    else:
        CT = rotorDict["CT"]        
    
    # If the rotor is active, return aerodynamic force.
    # Otherwise, return zero
    if rotorDict["active"]:
        F_wind_out = F_avg(rotorDict, V_10) + fRed(rotorDict, V_10)*(F_var(rotorDict, V_hub,  x_dot, CTVrel) - F_avg(rotorDict, V_10))
        return F_wind_out, CTVrel
    else:
        return 0., CTVrel
    
def F_var(rotorDict, V_hub, x_dot= 0., CT=0.):
    """ Calculate the time varying rotor forcing for a 10 minutes
    period
    """

    # relative velocity (wind minus structural)
    V_rel = V_hub-x_dot

    return 0.5*c["rho_air"]*rotorDict["ARotor"]*CT*V_rel*np.abs(V_rel)    



#should xdot beequal to zero. and ct the same ??????//

