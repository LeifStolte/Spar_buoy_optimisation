import numpy as np
import pylab as plt
import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'functions', 'python'))
sys.path.append(helpers_path)

from common import loadFromJSON, loadConstants,generateRandomPhases, Timer 
from waves import calculateFreeSurfaceElevationTimeSeries, calculateKinematics, calculateRegularWaveParameters 
from wind import calculateKaimalSpectrum,  calculateWindTimeSeriesFFT
from integration import ode4
from floaterIntegration import dqdt
from plotting import makeplots



of = "outputFig"
os.makedirs(of, exist_ok=True)
def ofy(fileName):
    return os.path.join(of, fileName)

plt.close('all')

#%% Q13: response to waves and steady wind

# Load the variables here

input_vars_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'assignment5/python','inputVariables'))
output_vars_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'outputVariables'))
time_json_path = os.path.join(input_vars_path, "time.json")
spar_json_path = os.path.join(output_vars_path, "SparBuoyDataComplete.json")
IEA22MWRotor_json_path = os.path.join(input_vars_path, "iea22mw.json")
waves_json_path = os.path.join(input_vars_path, "wave13.json")
wind_json_path = os.path.join(input_vars_path, "wind13.json")
timeInfo = loadFromJSON(time_json_path)
constants = loadConstants()
SparBuoyData = loadFromJSON(spar_json_path)






z = np.linspace(SparBuoyData["z_Bot"], 0., 100)
SparBuoyData["z"] = z
# Dry decay test
SparBuoyData["CD"] = 0.6

# Wave kinematics - should be zero
waves = loadFromJSON(waves_json_path);
waves["z"] = z


with Timer("wind"):
    # Wind speed - should be constant
    # FIXME Modify the input file wind13.json so that it represents a steady wind
    wind = loadFromJSON(wind_json_path);
    wind.update(timeInfo)
    wind["t"] = np.arange(0.,wind["TDur"] ,wind["dt"])
    wind = calculateKaimalSpectrum(wind)
    wind = generateRandomPhases(wind, 1)
    wind = calculateWindTimeSeriesFFT(wind)


# Load the rotor
IEA22MWRotor = loadFromJSON(IEA22MWRotor_json_path)
IEA22MWRotor["ARotor"] = 0.25*np.pi*IEA22MWRotor["DRotor"]**2

# Controller parameter & state of the rotor
IEA22MWRotor["gamma"] = 0.
IEA22MWRotor["active"] = True                         #why is it active

with Timer("waves"):
    # calculate the wave kinematics - zero at this stage
    waves.update(timeInfo)
    waves["t"] = np.arange(0.,wind["TDur"] ,wind["dt"])
    waves = calculateRegularWaveParameters(waves)
    waves = calculateFreeSurfaceElevationTimeSeries(waves)
    waves = calculateKinematics(waves)

# Integration time array
n = int((timeInfo["TDur"]-2*timeInfo["dt"])/(2*timeInfo["dt"]))+1
tode = np.linspace(0., timeInfo["TDur"]-2*timeInfo["dt"], n)

# Response
# Pitch decay with wind
with Timer("Integration"):
    # Initial conditions for surge decay
    q0 = np.array([0,0,0,0,np.nan])
    q = ode4(dqdt, tode, q0, SparBuoyData, IEA22MWRotor, waves, wind)

response = dict()
response["t"] = tode;
response["x1"] = q[:,0]
response["x5"] = q[:,1]

fig13 = makeplots(wind, waves, SparBuoyData, response, timeInfo, 'g');
fig13[0,0].figure.savefig(ofy("fig13.pdf"))
#%% Import the results from Q12
from main_q12 import response as response_waveonly
#fig13a = makeplots(wind, waves, SparBuoyData, response_waveonly, timeInfo, 'b', ax=fig13)
#fig13a[0,0].legend(["Wind Forcing", "No Wind Forcing"])
#fig13a[0,0].figure.savefig(ofy("fig13a.pdf"))


print(f'Q13 Surge Standard deviation [m]: {np.std(q[:,0])}');
print(f'Q13 Pitch Standard deviation [deg]: {np.rad2deg(np.std(q[:,1]))}')


# Calculate and print standard deviations after transient (TTrans)
post_transient = response["t"] > timeInfo["TTrans"]
surge_std_post = np.std(q[:, 0][post_transient])
pitch_std_post = np.rad2deg(np.std(q[:, 1][post_transient]))

print(f'Q13 Surge Std after TTrans [m]: {surge_std_post}')
print(f'Q13 Pitch Std after TTrans [deg]: {pitch_std_post}')


#what value for q0 conditions ???????????