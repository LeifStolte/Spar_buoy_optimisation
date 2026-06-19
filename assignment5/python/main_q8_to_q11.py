import numpy as np
import pylab as plt
import os
import sys


# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the function folder to the path
helpers_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'functions', 'python'))
sys.path.append(helpers_path)

from common import loadFromJSON, loadConstants
from integration import ode4
from floaterIntegration import dqdt
from plotting import makeplots


of = "outputFig"
os.makedirs(of, exist_ok=True)
def ofy(fileName):
    return os.path.join(of, fileName)

plt.close('all')

#%% Q10: DRY decays

# Load the variables here

input_vars_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'assignment5/python','inputVariables'))
output_vars_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'outputVariables'))
time_json_path = os.path.join(input_vars_path, "time.json")
spar_json_path = os.path.join(output_vars_path, "SparBuoyDataComplete.json")
IEA22MWRotor_json_path = os.path.join(input_vars_path, "iea22mw.json")

timeInfo = loadFromJSON(time_json_path)
constants = loadConstants()
SparBuoyData = loadFromJSON(spar_json_path)



z = np.linspace(SparBuoyData["z_Bot"], 0., 100)
SparBuoyData["z"] = z
SparBuoyData["CD"] = 0 


# Wave kinematics - should be zero
waves = loadFromJSON(os.path.join(input_vars_path, 'nowaves.json'));
waves["z"] = z

# Wind speed - should be zero
wind = loadFromJSON(os.path.join(input_vars_path, 'nowind.json'));
wind.update(timeInfo)
wind["t"] = np.arange(0.,wind["TDur"] ,wind["dt"])
wind["V_hub"] = np.zeros_like(wind["t"])

# Load the rotor
IEA22MWRotor = loadFromJSON(IEA22MWRotor_json_path)
IEA22MWRotor["ARotor"] = 0.25*np.pi*IEA22MWRotor["DRotor"]**2
# Controller parameter & state of the rotor
IEA22MWRotor["gamma"] = 0.
IEA22MWRotor["active"] = False

# calculate the wave kinematics - zero at this stage
waves.update(timeInfo)
waves["t"] = np.arange(0.,wind["TDur"] ,wind["dt"])
waves["u"] = np.zeros((len(waves["t"]), len(waves["z"])))
waves["ut"] = np.zeros_like(waves["u"])
waves["eta"] = np.zeros(len(waves["t"]))

# Integration time array
tode = np.arange(0., timeInfo["TDur"], 2*timeInfo["dt"])

# Initial conditions for surge decay
# FIXME: correct q0
# Initial conditions for surge decay
# FIXME: correct q0
q0 = np.array([1,0,0,0,np.nan])
q = ode4(dqdt, tode,q0, SparBuoyData, IEA22MWRotor, waves, wind)

response_surge = dict()
response_surge["t"] = tode;
response_surge["x1"] = q[:,0]
response_surge["x5"] = q[:,1]

fig10a = makeplots(wind, waves, SparBuoyData, response_surge, timeInfo, 'b');
fig10a[0,0].figure.savefig(ofy("fig10a.pdf"))

# Initial conditions for pitch decay
# FIXME: correct q0
q0 = np.array([0,0.1,0,0,np.nan])
q = ode4(dqdt, tode,q0, SparBuoyData, IEA22MWRotor, waves, wind)

response = dict()
response["t"] = tode;
response["x1"] = q[:,0]
response["x5"] = q[:,1]

fig10b = makeplots(wind, waves, SparBuoyData, response, timeInfo, 'b');
fig10b[0,0].figure.savefig(ofy("fig10b.pdf"))

#%% Q11: wet decays

SparBuoyData["CD"] = 0.6

# Initial conditions for surge decay
# FIXME: correct q0
q0 = np.array([1,0,0,0,np.nan])
q = ode4(dqdt, tode,q0, SparBuoyData, IEA22MWRotor, waves, wind)

response_hydron = dict()
response_hydron["t"] = tode;
response_hydron["x1"] = q[:,0]
response_hydron["x5"] = q[:,1]

fig11a = makeplots(wind, waves, SparBuoyData, response_hydron, timeInfo, 'g', ax=fig10a)
fig11a[0,0].legend(["no drag", "drag"])
fig11a[0,0].figure.savefig(ofy("fig11a.pdf"))

# Initial conditions for pitch decay
# FIXME: correct q0
q0 = np.array([0,0.1,0,0,np.nan])
q = ode4(dqdt, tode,q0, SparBuoyData, IEA22MWRotor, waves, wind)

response = dict()
response["t"] = tode;
response["x1"] = q[:,0]
response["x5"] = q[:,1]

fig11b = makeplots(wind, waves, SparBuoyData, response, timeInfo, 'g', ax=fig10b)
fig11b[0,0].legend(["no drag", "drag"])


fig11b[0,0].figure.savefig(ofy("fig11b.pdf"))

#%% Q11 Large decay test

SparBuoyData["CD"] = 0.

# Initial conditions for pitch decay
# FIXME: correct q0
q0 = np.array([0,1,0,0,np.nan])
q_nodrag = ode4(dqdt, tode,q0, SparBuoyData, IEA22MWRotor, waves, wind)

response_nodrag = dict()
response_nodrag["t"] = tode;
response_nodrag["x1"] = q_nodrag[:,0]
response_nodrag["x5"] = q_nodrag[:,1]

# Enable drag forcing
# FIXME: CD
SparBuoyData["CD"] = 0.6

# Initial conditions for pitch decay
# FIXME: correct q0
q0 = np.array([0,1,0,0,np.nan])
q_drag = ode4(dqdt, tode,q0, SparBuoyData, IEA22MWRotor, waves, wind)

response_drag = dict()
response_drag["t"] = tode;
response_drag["x1"] = q_drag[:,0]
response_drag["x5"] = q_drag[:,1]

fig11c = makeplots(wind, waves, SparBuoyData, response_nodrag, timeInfo, 'b');
fig11c = makeplots(wind, waves, SparBuoyData, response_drag, timeInfo, 'g', ax=fig11c);
fig11c[0,0].legend(["no drag", "drag"])
fig11c[0,0].figure.savefig(ofy("fig11c.pdf"))