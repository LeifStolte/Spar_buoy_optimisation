import numpy as np
from  monopile import *
from  rotor import *
from bisect import bisect_left as lookup

def rk4(f, h, y0, t0, *args, **kwargs):
    """
    h => dt 
    y0 => q0
    t0 => starting point for integration, from which we move with h steps
    args, kwargs => more arguments to the integrand function
    """
    k1 = f(t0, y0, *args, **kwargs)
    k2 = f(t0 + h / 2, y0 + h / 2 * k1, *args, **kwargs)
    k3 = f(t0 + h / 2, y0 + h / 2 * k2,*args, **kwargs)
    k4 = f(t0 + h, y0 + h * k3, *args, **kwargs)
    return y0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def ode4(odefun, tspan, y0, *args, **kwargs):
    """
    Python implementation of the rk4 integrator

    Parameters:
    ----------

    odefun: function handle
        Function to be integrated
    tspan: numpy array
        Time span
    y0: numpy array
        Initial conditions
    
    Returns:
    --------
    y : numpy array
        Integrated function

    """
    y = np.zeros((len(tspan),len(y0))) # array to store the solution
    y[0, :] = y0 # save the initial condition at t0
    h = np.diff(tspan) # time step array
    y_ini = y0 # store the initial condition for the first time step

    for i_, t_ in enumerate(tspan[:-1]):
        ti = tspan[i_]
        hi = h[i_]
        y_integrated = rk4(odefun, hi, y_ini, ti, *args, **kwargs)
        y_ini = y_integrated # save the initial condition for the next loop
        y[i_+1, :] = y_integrated # save the result in the solution array

    

    return y

def dqdt( t, q,
                structure,
                rotor,
                waves,
                wind,arg):

        alpha, alphaDot = q

        GF = GFCalc(t,
                        alphaDot,
                        structure,
                        rotor,
                        waves,
                        wind,arg)

        M = structure["GM"]
        D = structure["GD"]
        K = structure["GK"]

        # Monopile modal acceleration
        # FIXME Assignment 3 Q1.6: implement formula for alphaDotDot as a function of GF, GD, GK, GM
        
        
        
        # Calculate the expression step by step
        term1 = -D * alphaDot
        term2 = -K * alpha
        term3 = GF
        numerator = term1 + term2 + term3
        alphaDotDot = (1./M) * numerator
        
        
        dqdtOut = np.zeros(2)
        dqdtOut[0] = alphaDot  # First element is velocity
        dqdtOut[1] = alphaDotDot  # Second element is acceleration

        return dqdtOut

def GFCalc(t, alphaDot, structure, rotor, waves, wind, arg):    
    
    x_dot = alphaDot * structure["phiNodal"]
    
    # Wave contribution to Generalized Forcing
    wetPart = np.less_equal(structure["zBeamNodal"], 0.)    
    x_dot_submerged = x_dot[wetPart]
    phiNodalSubmerged = structure["phiNodal"][wetPart]
    
    i_ = lookup(waves["t"], t)
    u, ut, z = waves["u"][i_, :], waves["ut"][i_, :], waves["z"]
    df = forceDistributed(structure, u, ut, z, x_dot_submerged)
    
    # Calculate the generalized forcing from the waves
    GFWaves = np.trapz(df * phiNodalSubmerged, z)


    if arg == "onlyWaves":
        return GFWaves
    elif arg == "Total":
        # Wind contribution to Generalized Forcing
        i_ = lookup(wind["t"], t)
        V_10, V_hub = wind["V_10"], wind["V_hub"][i_]
        phiHub = structure["phiNodal"][-1]
        
        # Compute the velocity of the hub due to structural deformation
        x_dot_rotor = alphaDot * phiHub
        
        # Calculate the generalized forcing from the wind
        F_aero = F_wind(rotor, V_10, V_hub, x_dot_rotor)
        GFWind = F_aero * phiHub
        
        # Total generalized force (ensure it's a scalar)
        GF = float(GFWind + GFWaves)
    
    return GF