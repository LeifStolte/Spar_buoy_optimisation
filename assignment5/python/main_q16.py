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
from waves import calculateFreeSurfaceElevationTimeSeriesFFT, calculateKinematicsFFT, calculateJONSWAPSpectrum
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
#load functions 
input_vars_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'assignment5/python','inputVariables'))
output_vars_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'outputVariables'))
time_json_path = os.path.join(input_vars_path, "time.json")
spar_json_path = os.path.join(output_vars_path, "SparBuoyDataComplete.json")
IEA22MWRotor_json_path = os.path.join(input_vars_path, "iea22mw.json")
waves_json_path = os.path.join(input_vars_path, "wave13.json")
wind_json_path_a = os.path.join(input_vars_path, "wind16A.json")
wind_json_path_b = os.path.join(input_vars_path, "wind16B.json")
timeInfo = loadFromJSON(time_json_path)
constants = loadConstants()
SparBuoyData = loadFromJSON(spar_json_path)

# %% ----Q16: RESPONSE TO WIND--------------------------------------


# Load the time information
timeInfo = loadFromJSON(time_json_path)

# Wind speed - should be constant
# Just a loop to handle both wind cases
wind_a = loadFromJSON(wind_json_path_a)
wind_b = loadFromJSON(wind_json_path_b)
winds = [wind_a, wind_b]
colors = ['b', 'g']
labels = ['10 m/s', '16 m/s']

# Initialize the figure outside the loop
#fig16 = plt.figure()
#plt.hold(True)  # Keep the plot open for multiple plots
# Create figure and axes once
fig16, ax16 = plt.subplots(4,2, sharex='col')

for i in range(len(winds)):
    wind = winds[i]
    wind.update(timeInfo)
    wind['t'] = np.arange(0, wind['TDur'], wind['dt'])
    wind["V_hub"] = np.zeros_like(wind["t"])
    wind = calculateKaimalSpectrum(wind)
    wind = generateRandomPhases(wind, 2)  
    wind = calculateWindTimeSeriesFFT(wind)

    # Load the rotor and other necessary parameters
    IEA22MWRotor = loadFromJSON(IEA22MWRotor_json_path)
    IEA22MWRotor['ARotor'] = 0.25 * np.pi * IEA22MWRotor['DRotor']**2
    IEA22MWRotor['gamma'] = 0.0
    IEA22MWRotor['active'] = True

    # Disable drag forcing
    SparBuoyData = loadFromJSON(spar_json_path)
    SparBuoyData['CD'] = 0  # Dry decay test

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
    waves['eta'] = np.zeros_like(waves['t'])        # free surface elevation
    waves['u']   = np.zeros((len(waves['t']), len(waves['z'])))  # horizontal velocity at each z
    waves['ut']  = np.zeros((len(waves['t']), len(waves['z'])))  # vertical velocity at each z


    # Response
    tode = np.arange(0., timeInfo["TDur"], 2 * timeInfo["dt"])
    q0 = np.array([0, 0, 0, 0, np.nan])
    q = ode4(dqdt, tode, q0, SparBuoyData, IEA22MWRotor, waves, wind)

    response = dict()
    response["t"] = tode
    response["x1"] = q[:, 0]
    response["x5"] = q[:, 1]

    # Plot the results on the same figure
    fig16 = makeplots(wind, waves, SparBuoyData, response, timeInfo, colors[i], ax=ax16)
    
    fig16[0,0].legend(["10 [m/s]", "16 [m/s]"])
    plt.savefig(ofy("fig16.pdf"))
    
    # Print the standard deviations
    print('--- Results for wind speed:', labels[i], '---')
    print(f'Q16 Surge Standard deviation [m]: {np.std(q[:,0])}')
    print(f'Q16 Pitch Standard deviation [deg]: {np.rad2deg(np.std(q[:,1]))}')
    # Calculate and print standard deviations after transient (TTrans)
    post_transient = response["t"] > timeInfo["TTrans"]
    surge_std_post = np.std(q[:, 0][post_transient])
    pitch_std_post = np.rad2deg(np.std(q[:, 1][post_transient]))

    print(f'Q16 Surge Std after TTrans [m]: {surge_std_post}')
    print(f'Q16 Pitch Std after TTrans [deg]: {pitch_std_post}')
# Add labels and legend
plt.legend(labels)  # Use labels from the loop
