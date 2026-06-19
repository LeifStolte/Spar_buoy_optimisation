"""
Main script for offshore wind energy assignment 5.
Loads input variables, performs preliminary computations, and saves results.
"""

import os
import sys
import numpy as np

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'functions', 'python'))
sys.path.append(helpers_path)



input_vars_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'assignment5/python','inputVariables'))
time_json_path = os.path.join(input_vars_path, "time.json")
spar_json_path = os.path.join(input_vars_path, "SparBuoyData.json")
IEA22MWRotor_json_path = os.path.join(input_vars_path, "iea22mw.json")

from common import loadFromJSON, saveToJSON, loadConstants
from table import print_all_tables

timeInfo = loadFromJSON(time_json_path)
SparBuoyData = loadFromJSON(spar_json_path)
constants = loadConstants()

g = constants['g']
rhow = constants['rho_water']
rhoa = constants['rho_air']
rhos = SparBuoyData['rho_Steel']
mtu = SparBuoyData['M_Turbine']
zturb = SparBuoyData['z_CM_Turbine']
zhub = SparBuoyData['z_Hub']
fb = SparBuoyData['fb']
draft = SparBuoyData['draft']
Dspar = SparBuoyData['DMonopile']
th = SparBuoyData['Thickness']
Dhp = Dspar
Kmoor = SparBuoyData['K_Moor']
zmoor = SparBuoyData['z_Moor']
Cm = SparBuoyData['CM'] - 1.0;
Cd = SparBuoyData['CD']
mt = SparBuoyData['M_Tower']
zCMt = SparBuoyData['z_CM_Tower']
ICMt = SparBuoyData['I_CM_Tower']
BallastHeight = SparBuoyData['BallastHeightindraft']
BallastCOG = SparBuoyData['Ballast_COG']
mb = SparBuoyData['M_Ballast']


IEA22MWRotor = loadFromJSON(IEA22MWRotor_json_path)

B11 = SparBuoyData["B11"]; 
thrustr = 2*SparBuoyData["MaxThrust"]

##% Preliminary computations

# Calculate center of buoyancy, center of mass, floater inertias
# Location of floater bottom
zbot = -draft
# Location of ballast center of gravity


zballst = BallastCOG # height of draft is ballast 

# FIXME: Center of buoyancy
zCB = zbot/2


# FIXME: Displacement volume
Vol = (Dspar/2)**2 * np.pi * draft

# FIXME: Spar length
ls = draft + fb

# FIXME: Spar mass without ballast
ms = ((Dspar/2)**2* np.pi - ((Dspar-2*th)/2)**2* np.pi)  * rhos * ls


# Floater mass with ballast
mf = ms + mb


# FIXME: Spar center of mass without ballast
zCMs = fb-ls/2

# FIXME: Floater center of mass with ballast
zCMf = (zCMs * ms + zballst * mb)/mf


# FIXME: Spar inertia about its Center of Mass without ballast

ICMs = 1/12 * ms * (3*((Dspar-Dspar-2*th)/2)**2+ls**2)

# FIXME: Floater inertia about floater CM with ballast
ICMf = 1/12 * mf * (3*((Dspar/2-(Dspar-2*th)/2))**2+ls**2)



##% Q1: System matrices

# FIXME: Total mass
mtot = mf + mt + mtu

# FIXME: Total center of mass
zCMtot = (zCMf * mf + zCMt * mt + zturb *mtu)/mtot


# FIXME: Total inertia about flotation point

I_Spar = ICMs +ms * (zCMs**2)
I_tower = ICMt + mt * (zCMt**2)
I0_turbine = mtu * (zturb**2)
I0_ballast = mb * (zballst**2)

IOtot = I_Spar + I_tower + I0_ballast + I0_turbine



# MASS MATRIX
# Total system mass and inertia matrix
# M1: total mass
# M15, M51: mass times center of mass (coupling surge-pitch)
# M5: total inertia about flotation point
M1 = mtot
M15 = mtot * zCMtot
M51 = mtot * zCMtot
M5 = IOtot

M = np.array([[M1 , M15], [M51, M5]])

# ADDED MASS MATRIX
# Added mass matrix (hydrodynamic inertia due to water acceleration)
# A1: surge added mass (integral over draft)
# A15, A51: coupling terms (integral of z over draft),A15 and A51 are set equal because the added mass matrix 
# is symmetric for a vertical cylinder in water.
# A5: pitch added mass (integral of z^2 over draft)

A1 = (np.pi/4) * rhow * Dspar**2 * Cm * draft
A15 = -(np.pi/4) * rhow * Dspar**2 * Cm * draft**2 / 2
A51 = A15
A5 = (np.pi/4) * rhow * Dspar**2 * Cm * draft**3 / 3
A = np.array([[A1, A15], [A51, A5]])

# Damping MATRIX
# Damping matrix (linearized hydrodynamic drag)
B = np.array([[B11, 0],[0 ,0]])

# FIXME: Water Plane Inertia
# Water plane inertia (used for hydrostatic restoring)
IAA = np.pi*Dspar**4/64
print(IAA)

# FIXME: hydrodynamic stiffness
# Hydrodynamic stiffness matrix (placeholder, typically from hydrostatic restoring)
Cs1 = 0
Cs15 = 0
Cs51 = 0
Cs5 = rhow * g * IAA + mtot * g * (zCB - zCMtot)




Chst = np.array([[Cs1,Cs15], [Cs51, Cs5]])

# Mooring restoring matrix ???
# Mooring restoring matrix (from mooring line spring force and moment)
# Uses zmoor (vertical position of mooring attachment)
Cm1 = Kmoor 
Cm15 = Kmoor * zmoor
Cm51 = Kmoor * zmoor
Cm5 = Kmoor * zmoor**2
Cmoor = np.array([[Cm1, Cm15], [Cm51, Cm5]])


# Total restoring matrix (hydrostatic + mooring)
C = Chst + Cmoor

# Add preliminary computation variables to SparBuoyData for table output
SparBuoyData["zCB"] = zCB
SparBuoyData["ls"] = ls
SparBuoyData["ms"] = ms
SparBuoyData["mf"] = mf
SparBuoyData["zCMs"] = zCMs
SparBuoyData["zCMf"] = zCMf
SparBuoyData["ICMs"] = ICMs
SparBuoyData["ICMf"] = ICMf
SparBuoyData["IAA"] = IAA
SparBuoyData["MTot"] = mtot
SparBuoyData["zCM_Tot"] = zCMtot
SparBuoyData["IO_Tot"] = IOtot

SparBuoyData["M"] = M
SparBuoyData["C"] = C
SparBuoyData["A"] = A
SparBuoyData["B"] = B

##% Natural Frequencies

# FIXME: calculate C over MA
CoMA = C @ np.linalg.inv(M + A)
eigVal, eigVec = np.linalg.eig(CoMA)


# Natural frequencies
omeganat = np.sqrt(eigVal)
fnat = omeganat/2/np.pi

SparBuoyData["fnat"] = fnat

# Natural periods
Tnat = 1./fnat



# FIXME: Added mass in heave
# A33 = ...; 
a33 = 0.5

# FIXME: Hydrostatic restoring in heave
c33 = 1

# FIXME: Heave natural period
Theave = 1


os.makedirs("outputVariables", exist_ok=True)
saveToJSON(SparBuoyData, "outputVariables/SparBuoyDataComplete.json")

# Print comprehensive tables of all calculations

print_all_tables("outputVariables/SparBuoyDataComplete.json")