'''
Filename: c:\\Users\\fabpi\\modules\\46211assignmentsolution\\src\\assignmentcode\\edition2024\\python\\runner.py
Path: c:\\Users\\fabpi\\modules\\46211assignmentsolution\\src\\assignmentcode\\edition2024\\python
Created Date: Saturday, October 12th 2024, 8:27:19 am
Author: Fabio Pierella

Copyright (c) 2024 DTU Wind and Energy Systems
'''

from waves import *
from wind import *
from common import *
from monopile import *
from rotor import *
from integration import ode4, dqdt
from loads import *

def runEnvironmentalCondition(wind, waves, rotor, monopile, timeInfo):
    
    # Load time info and calculate time vector
    waves.update(timeInfo); wind.update(timeInfo)
    waves["t"] = np.arange(0., waves["TDur"], waves["dt"])
    wind["t"] = np.arange(0., wind["TDur"], wind["dt"])
    
    # Calculate wind 
    wind = calculateKaimalSpectrum(wind)
    wind = generateRandomPhases(wind, seed=wind["randomSeed"])
    # Note: replace with slow version if the FFT one does not work
    wind = calculateWindTimeSeriesFFT(wind)
    
    # Calcualte waves
    waves = calculateJONSWAPSpectrum(waves)
    waves = generateRandomPhases(waves, seed=waves["randomSeed"])
    # Note: replace with slow version if the FFT one does not work
    waves = calculateFreeSurfaceElevationTimeSeriesFFT(waves)
    waves = calculateKinematicsFFT(waves)
            
    # Calculate response    
    q0 = [0., 0.]
    tIntegration = np.arange(0., timeInfo["TDur"], 2*timeInfo["dt"])
    q = ode4(dqdt, tIntegration, q0, monopile, rotor,
                waves, wind)
    
    # Save it into dict
    response = dict()
    response["t"] = tIntegration
    response["alphaDot"] = q[:,1]
    response["alphaDotDot"] = np.gradient(q[:,1], tIntegration)
    
    # Calculate the moments
    windDownsampled = downsample(wind, dropEvery=2, listOfFields=["t", "V_hub"])
    windLoads = calculateStaticWindLoads(windDownsampled, rotor, 
                                    monopile, response)

    # Calculate the static wave loads
    wavesDownsampled = downsample(waves, dropEvery=2, listOfFields=["t", "u", "ut", "eta"])
    waveLoads = calculateStaticWaveLoads(wavesDownsampled, monopile, response)

    # Calculate the dynamic loads
    monopile = computeElementwiseQuantities(monopile)
    dynamicLoads = calculateDynamicLoads(monopile, response)
    
    outputDict = {}
    outputDict["waves"] = waveLoads
    outputDict["wind"] = windLoads
    outputDict["dynamic"] = dynamicLoads
    
    # Compute the total overturning moment
    outputDict["total"] = dict()
    outputDict["total"]["t"] = waveLoads["t"].copy()
    outputDict["total"]["F"] = waveLoads["F"] + windLoads["F"] + dynamicLoads["F"]
    outputDict["total"]["M"] = waveLoads["M"] + windLoads["M"] + dynamicLoads["M"]
    
    return outputDict