import numpy as np
import pylab as plt
import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'functions', 'python'))
sys.path.append(helpers_path)

from common import loadFromJSON, loadConstants, generateRandomPhases
from waves import calculateJONSWAPSpectrum, calculateFreeSurfaceElevationTimeSeriesFFT, calculateKinematicsFFT
from wind import calculateKaimalSpectrum,  calculateWindTimeSeriesFFT, calculateWindTimeSeries
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

# %% ----Q18: RESPONSE TO GAMMA PARAMETERS--------------------------------------

# Load the time information
input_vars_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'assignment5/python','inputVariables'))
output_vars_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'outputVariables'))
spar_json_path = os.path.join(output_vars_path, "SparBuoyDataComplete.json")
IEA22MWRotor_json_path = os.path.join(input_vars_path, "iea22mw.json")
waves_json_path = os.path.join(input_vars_path, "wave13.json")
wind_json_path = os.path.join(input_vars_path, "wind20.json")
time_json_path = os.path.join(input_vars_path, "time.json")
timeInfo = loadFromJSON(time_json_path)

# Wind speed - should be constant
# Just a loop to handle both wind cases
wind = loadFromJSON(wind_json_path)
winds = [wind, wind, wind, wind]
colors = ['b', 'g', 'r', 'm']

# Gamma values for the four cases
gammas = [0.1, 0.25, 0.5, 0.9]
labels = ['Inst. control gamma 0.1', 'Inst. control gamma 0.25', 'Inst. control gamma 0.5', 'Inst. control gamma 0.9']

# Initialize the figure outside the loop
fig18, ax18 = plt.subplots(4,2, sharex='col')  # fig17 is the figure
#plt.hold(True)  # Keep the plot open for multiple plots

labels = [f'Gamma={g}' for g in gammas]



# Automate overlay plot for each gamma value
for i, gamma in enumerate(gammas):
    colors = ['k', 'b', 'g']
    labels_overlay = ['No wind gusts, no waves', 'Waves, no wind', f'Waves, wind (gamma={gamma})']
    fig, ax = plt.subplots(4,2, sharex='col')

    SparBuoyData = loadFromJSON(spar_json_path)
    SparBuoyData['z'] = np.linspace(SparBuoyData['z_Bot'], 0, 100)
    IEA22MWRotor = loadFromJSON(IEA22MWRotor_json_path)
    tode = np.arange(0., timeInfo["TDur"], 2 * timeInfo["dt"])
    q0 = np.array([0, 0, 0, 0, 0])
    IEA22MWRotor['ARotor'] = 0.25 * np.pi * IEA22MWRotor['DRotor']**2
    IEA22MWRotor['gamma'] = 0.0
    IEA22MWRotor['active'] = True

    # 1. Waves, no wind
    wind = loadFromJSON(wind_json_path)
    wind.update(timeInfo)
    wind['t'] = np.arange(0, wind['TDur'], wind['dt'])
    wind['V_hub'] = np.zeros_like(wind['t'])
    waves = loadFromJSON(waves_json_path)
    waves['z'] = SparBuoyData['z']
    waves.update(timeInfo)
    waves['t'] = np.arange(0, waves['TDur'], waves['dt'])
    waves = calculateJONSWAPSpectrum(waves)
    waves = generateRandomPhases(waves, 1)
    waves = calculateFreeSurfaceElevationTimeSeriesFFT(waves)
    waves = calculateKinematicsFFT(waves)
    IEA22MWRotor['gamma'] = 0.0
    q = ode4(dqdt, tode, q0, SparBuoyData, IEA22MWRotor, waves, wind, 'q20')
    response = dict()
    response["t"] = tode
    response["x1"] = q[:, 0]
    response["x5"] = q[:, 1]
    makeplots(wind, waves, SparBuoyData, response, timeInfo, colors[1], ax=ax, label=labels_overlay[1])

    # 1. No wind, no waves
    
    SparBuoyData['CD'] = 0
    wind = loadFromJSON(wind_json_path)
    wind.update(timeInfo)
    wind['t'] = np.arange(0, wind['TDur'], wind['dt'])
    wind['V_hub'] = np.zeros_like(wind['t'])
    waves = loadFromJSON(waves_json_path)
    waves['z'] = SparBuoyData['z']
    waves.update(timeInfo)
    waves['t'] = np.arange(0, waves['TDur'], waves['dt'])
    waves['eta'] = np.zeros_like(waves['t'])
    waves['u'] = np.zeros((len(waves['t']), len(waves['z'])))
    waves['ut'] = np.zeros((len(waves['t']), len(waves['z'])))
    
    
    
    
    q = ode4(dqdt, tode, q0, SparBuoyData, IEA22MWRotor, waves, wind, 'q20')
    response = dict()
    response["t"] = tode
    response["x1"] = q[:, 0]
    response["x5"] = q[:, 1]
    makeplots(wind, waves, SparBuoyData, response, timeInfo, colors[0], ax=ax, label=labels_overlay[0])

    # 3. Waves, wind (with current gamma)
    wind = loadFromJSON(wind_json_path)
    wind.update(timeInfo)
    wind['t'] = np.arange(0, wind['TDur'], wind['dt'])
    wind = calculateKaimalSpectrum(wind)
    wind = generateRandomPhases(wind, 2)
    wind = calculateWindTimeSeriesFFT(wind)
    waves = loadFromJSON(waves_json_path)
    waves['z'] = SparBuoyData['z']
    waves.update(timeInfo)
    waves['t'] = np.arange(0, waves['TDur'], waves['dt'])
    waves = calculateJONSWAPSpectrum(waves)
    waves = generateRandomPhases(waves, 1)
    waves = calculateFreeSurfaceElevationTimeSeriesFFT(waves)
    waves = calculateKinematicsFFT(waves)
    IEA22MWRotor['gamma'] = gamma
    q = ode4(dqdt, tode, q0, SparBuoyData, IEA22MWRotor, waves, wind, 'q20')
    response = dict()
    response["t"] = tode
    response["x1"] = q[:, 0]
    response["x5"] = q[:, 1]
    makeplots(wind, waves, SparBuoyData, response, timeInfo, colors[2], ax=ax, label=labels_overlay[2])

    
    plt.tight_layout()
    ax[0,0].figure.savefig(ofy(f"fig_var_compare_gamma_{gamma}.pdf"))