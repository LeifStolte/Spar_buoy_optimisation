import numpy as np
import pylab as plt
import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'functions', 'python'))
sys.path.append(helpers_path)

from common import loadFromJSON, loadConstants,generateRandomPhases
from waves import calculateJONSWAPSpectrum, calculateFreeSurfaceElevationTimeSeriesFFT, calculateKinematicsFFT
from wind import calculateKaimalSpectrum,  calculateWindTimeSeriesFFT
from integration import ode4
from floaterIntegration import dqdt
from plotting import makeplots

# Output folder setup
of = "outputFig"
os.makedirs(of, exist_ok=True)

def ofy(fileName):
    return os.path.join(of, fileName)

# Close any existing plots
plt.close('all')

# %% ----Q13: RESPONSE TO IRREGULAR WAVES--------------------------------------

# Load the variables here

input_vars_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'assignment5/python','inputVariables'))
output_vars_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'outputVariables'))
time_json_path = os.path.join(input_vars_path, "time.json")
spar_json_path = os.path.join(output_vars_path, "SparBuoyDataComplete.json")
IEA22MWRotor_json_path = os.path.join(input_vars_path, "iea22mw.json")
waves_json_path = os.path.join(input_vars_path, "wave13.json")
wind_json_path = os.path.join(input_vars_path, "wind15.json")
timeInfo = loadFromJSON(time_json_path)
constants = loadConstants()
SparBuoyData = loadFromJSON(spar_json_path)
# Load the rotor data
IEA22MWRotor = loadFromJSON(IEA22MWRotor_json_path)



# Controller parameter & state of the rotor
IEA22MWRotor['ARotor'] = 0.25 * np.pi * IEA22MWRotor['DRotor']**2
IEA22MWRotor['gamma'] = 0.0
IEA22MWRotor['active'] = True

# Wind speed - should be constant
# We set up a spectrum with zero turbulence.
# Just an easy way to reuse existing functions
random_seed_wind = 2
wind = loadFromJSON(wind_json_path)
wind.update(timeInfo)
wind['t'] = np.arange(0, wind['TDur'], wind['dt'])
wind["V_hub"] = np.zeros_like(wind["t"])

# fixme: Add WIND calculations here (use 3 functions if applicable)

wind.update(timeInfo)
wind["t"] = np.arange(0.,wind["TDur"] ,wind["dt"])
wind = calculateKaimalSpectrum(wind)
wind = generateRandomPhases(wind, random_seed_wind)
wind = calculateWindTimeSeriesFFT(wind)

# Vertical locations along floater
z = np.linspace(SparBuoyData['z_Bot'], 0, 100)
SparBuoyData['z'] = z

# Calculate the wave kinematics from Irregular wave
waves = loadFromJSON(waves_json_path)
waves['z'] = z
random_seed_waves = 1
waves.update(timeInfo)
waves['t'] = np.arange(0, waves['TDur'], waves['dt'])

# fixme: Add wave calculations here (use 4 functions if applicable)
waves = calculateJONSWAPSpectrum(waves)
waves = generateRandomPhases(waves, random_seed_waves)
waves = calculateFreeSurfaceElevationTimeSeriesFFT(waves)      # free surface elevation
waves  = calculateKinematicsFFT(waves) # horizontal velocity at each z

# Response
tode = np.arange(0., timeInfo["TDur"], 2 * timeInfo["dt"])
q0 = np.array([0, 0, 0, 0, np.nan])
q = ode4(dqdt, tode, q0, SparBuoyData, IEA22MWRotor, waves, wind)

response = dict()
response["t"] = tode
response["x1"] = q[:, 0]
response["x5"] = q[:, 1]

# Plotting results
fig15 = makeplots(wind, waves, SparBuoyData, response, timeInfo, 'b')
plt.savefig(ofy("fig15.pdf"))

# Calculate and print standard deviations
surge_std = np.std(q[:, 0])
pitch_std = np.rad2deg(np.std(q[:, 1]))

print(f'Q15 Surge Standard deviation [m]: {surge_std}')
print(f'Q15 Pitch Standard deviation [deg]: {pitch_std}')

# Calculate and print standard deviations after transient (TTrans)
post_transient = response["t"] > timeInfo["TTrans"]
surge_std_post = np.std(q[:, 0][post_transient])
pitch_std_post = np.rad2deg(np.std(q[:, 1][post_transient]))

print(f'Q15 Surge Std after TTrans [m]: {surge_std_post}')
print(f'Q15 Pitch Std after TTrans [deg]: {pitch_std_post}')



