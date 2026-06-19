import json
import time

import numpy as np
from scipy.optimize import root_scalar

from models import Constants


def loadFromJSON(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = np.array(value)

    return data


def saveToJSON(inputDictionary, file_path):
    if hasattr(inputDictionary, "to_mapping"):
        inputDictionary = inputDictionary.to_mapping()
    elif hasattr(inputDictionary, "__dict__") and not isinstance(inputDictionary, dict):
        inputDictionary = dict(vars(inputDictionary))

    def convert(obj):
        if hasattr(obj, "to_mapping"):
            return obj.to_mapping()
        if hasattr(obj, "__dict__") and not isinstance(obj, dict):
            return dict(vars(obj))
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(inputDictionary, file, default=convert)


def loadConstants():
    return Constants(g=9.81, rho_air=1.22, rho_water=1025.0)


def dispersion(f, h):
    omega = 2 * np.pi * f
    k = np.zeros_like(f)
    g = loadConstants().g

    myFun = lambda k, omega, g, h: omega ** 2 - g * k
    myFunPrime = lambda k, omega, g, h: -g  # use this one for the full dispersion relation: -g*(k*h*(1-np.tanh(k*h)**2)+np.tanh(k*h));

    k_guess = omega[0] / np.sqrt(g * h)

    for idx, freq in enumerate(omega):
        k[idx] = root_scalar(
            lambda x, freq=freq: myFun(x, freq, g, h),
            fprime=lambda x, freq=freq: myFunPrime(x, freq, g, h),
            x0=k_guess,
            method='newton',
        ).root
        k_guess = k[idx]

    return k


def lcg(seed, a=1103515245, c=12345, m=2**31, n=1):
    # Linear congruential random number generator.
    # Chosen to have the sampe implementation in Matlab and Python.
    # See: https://en.wikipedia.org/wiki/Linear_congruential_generator

    numbers = np.ones(n)
    numbers *= seed
    for i in range(1, n):
        numbers[i] = (a * numbers[i - 1] + c) % m
    return 2 * np.pi * np.array([x / m for x in numbers])  # Normalize to [0, 1]


def generateRandomPhases(inputDict, seed=2):
    # Stating a seed will allow for repeatability
    # of your randomness
    # rng = np.random.default_rng(seed)
    # phi = 2*np.pi*rng.random(len(inputDict["Spectrum"]))
    spectrum = getattr(inputDict, "Spectrum", None)
    if spectrum is None and isinstance(inputDict, dict):
        spectrum = inputDict["Spectrum"]
    phi = lcg(seed, n=len(spectrum))

    if hasattr(inputDict, "copy_with"):
        return inputDict.copy_with(randomPhases=phi)
    outputDict = inputDict.__class__(inputDict)
    outputDict["randomPhases"] = phi
    return outputDict


def downsample(inputDict, dropEvery=2, listOfFields=None):
    if hasattr(inputDict, "copy_with"):
        outputDict = inputDict.copy()
        if listOfFields is None:
            listOfFields = [name for name in vars(inputDict).keys() if getattr(inputDict, name, None) is not None]
        for field_ in listOfFields:
            value = getattr(inputDict, field_)
            if value is not None:
                setattr(outputDict, field_, value[::dropEvery])
        if hasattr(outputDict, "dt") and outputDict.dt is not None:
            outputDict.dt *= dropEvery
        return outputDict

    if listOfFields is None:
        listOfFields = inputDict.keys()  # all of them

    outputDict = dict()
    outputDict.update(inputDict)  # copy everything
    for field_ in listOfFields:
        outputDict[field_] = inputDict[field_][::dropEvery]
    outputDict["dt"] *= dropEvery
    return outputDict


def pad2(vector, size):
    # Padding an array with zeros up to size
    return np.pad(vector, [1, size - len(vector) - 1])


class Timer(object):
    """This is just a simple timer as we don't have tic - toc in Python.
    Taken from here: https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions
    """

    def __init__(self, name=None):
        self.name = name
        self.tstart = None

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, exc_type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        elapsedTime = time.time() - self.tstart
        print('Elapsed: %.5f' % (elapsedTime))
