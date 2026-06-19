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

# %% ----Q17: RESPONSE TO PITCH INSTABILITY--------------------------------------

# Load the time information
input_vars_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'assignment5/python','inputVariables'))
output_vars_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'outputVariables'))
spar_json_path = os.path.join(output_vars_path, "SparBuoyDataComplete.json")
IEA22MWRotor_json_path = os.path.join(input_vars_path, "iea22mw.json")
waves_json_path = os.path.join(input_vars_path, "wave13.json")
wind_json_path = os.path.join(input_vars_path, "wind16B.json")
time_json_path = os.path.join(input_vars_path, "time.json")
timeInfo = loadFromJSON(time_json_path)

# Wind speed - should be constant
# Just a loop to handle both wind cases

wind = loadFromJSON(wind_json_path)
winds = [wind, wind]
colors = ['b', 'g']

# Gamma values for the two cases
gammas = [0, 2]     # ARE GAMMAS BOTH 2 ???
labels = ['Inst. control gamma 0', 'Inst. control gamma 2']

# Initialize the figure outside the loop
fig17, ax17 = plt.subplots(4,2, sharex='col')  # fig17 is the figure
#plt.hold(True)  # Keep the plot open for multiple plots

for i in range(len(winds)):
    wind = winds[i]
    wind.update(timeInfo)
    wind['t'] = np.arange(0, wind['TDur'], wind['dt'])
    wind["V_hub"] = np.zeros_like(wind["t"])
    wind = calculateKaimalSpectrum(wind)  # Ensure this function is defined
    wind = generateRandomPhases(wind, 2)  # Ensure this function is defined
    wind = calculateWindTimeSeriesFFT(wind)
    
    # Load the rotor and other necessary parameters
    IEA22MWRotor = loadFromJSON(os.path.join(IEA22MWRotor_json_path))
    IEA22MWRotor['ARotor'] = 0.25 * np.pi * IEA22MWRotor['DRotor']**2
    IEA22MWRotor['gamma'] = gammas[i]
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
    waves['u'] = np.zeros((len(waves['t']), len(waves['z'])))
    waves['ut'] = np.zeros((len(waves['t']), len(waves['z'])))
    waves['eta'] = np.zeros_like(waves['t']) 

    # Response
    tode = np.arange(0., timeInfo["TDur"], 2 * timeInfo["dt"])
    q0 = np.array([0, 0, 0, 0, 0])
    q = ode4(dqdt, tode, q0, SparBuoyData, IEA22MWRotor, waves, wind, 'q17' )

    response = dict()
    response["t"] = tode
    response["x1"] = q[:, 0]
    response["x5"] = q[:, 1]

    # Plot the results on the same figure
    fig17 = makeplots(wind, waves, SparBuoyData, response, timeInfo, colors[i], ax=ax17)
    fig17[0,0].legend(["gamma 0", "gamma 2"])
    plt.savefig(ofy("fig17.pdf"))
    
    # Print the standard deviations
    print('--- Results for wind speed:', labels[i], '---')
    print(f'Q17 Surge Standard deviation [m]: {np.std(q[:,0])}')
    print(f'Q17 Pitch Standard deviation [deg]: {np.rad2deg(np.std(q[:,1]))}')
    # Calculate and print standard deviations after transient (TTrans)
    post_transient = response["t"] > timeInfo["TTrans"]
    surge_std_post = np.std(q[:, 0][post_transient])
    pitch_std_post = np.rad2deg(np.std(q[:, 1][post_transient]))

    print(f'Q17 Surge Std after TTrans [m]: {surge_std_post}')
    print(f'Q17 Pitch Std after TTrans [deg]: {pitch_std_post}')




# %% ----Q15: RESPONSE TO IRREGULAR WAVES & WIND--------------------------------
