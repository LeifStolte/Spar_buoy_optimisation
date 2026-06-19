import numpy as np
from common import loadConstants

g = loadConstants()["g"]
rho_water = loadConstants()["rho_water"]

def forceIntegrate(monopileDict, u, ut, z, x_dot):
    
    h = np.abs(z[0])
    u = u - x_dot
    
    df = forceDistributed(monopileDict, u, ut, z, x_dot)

   
    F = np.trapz(df, z)
    # FIXME Assignment 1 Q6:  
    # Calculate the moment as well
    
    dM = df*(h + z) # get zphys from wavesQ@ dictionary
    M = np.trapz(dM, z)
    
    return F, M

def forceDistributed(monopileDict, u, ut, z, x_dot):
    
    u = u - x_dot
    
    # FIXME Assignment 1 Q2.3:  add back the inertia forces
    df_drag = 0.5*rho_water*monopileDict["DMonopile"]*monopileDict["CD"]*np.abs(u)*u   # drag force
    df_inertia = rho_water * monopileDict["CM"] * (np.pi/4)*monopileDict["DMonopile"]**2 * ut   # acceleration
    df = df_drag + df_inertia

    return  df

def computeElementwiseQuantities(monopileDict):
    
    outputDict = dict()
    outputDict.update(monopileDict)
    
    # Compute missing element properties
    z = monopileDict["zBeamNodal"]
    dz = np.diff(z)
    outputDict["zBeamElement"] = z[:-1] + dz/2
    outputDict["dz"] = dz
    
    # Compute the phiNodal
    phi = monopileDict["phiNodal"]
    dPhi = np.diff(phi)
    outputDict["phiElement"] = outputDict["phiNodal"][:-1] + dPhi/2
    
    return outputDict
    