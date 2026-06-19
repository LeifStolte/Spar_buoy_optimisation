import time
import json
import numpy as np
from scipy.optimize import root_scalar

def loadFromJSON(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = np.array(value)
    
    return data
        
def saveToJSON(inputDictionary, file_path):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)
    
    with open(file_path, 'w') as file:
        json.dump(inputDictionary, file, default=convert)

def loadConstants():
    
    constants = dict()
    constants["g"] = 9.81;
    constants["rho_air"] = 1.22
    constants["rho_water"] = 1025.
    
    return constants

def dispersion(f, h):
    N = len(f)
    omega = 2*np.pi*f
    k = np.zeros_like(f)
    g = loadConstants()["g"]

    # FIXME: Assignment 1 Part 1 Q1
    myFun = lambda k, omega, g, h: omega**2 - g*k
    myFunPrime = lambda k, omega, g, h: -g  # use this one for the full dispersion relation: -g*(k*h*(1-np.tanh(k*h)**2)+np.tanh(k*h));

    kGuess = omega[0] / np.sqrt(g*h)

    for j in range(N):
        k[j] = root_scalar(lambda x: myFun(x,omega[j],g,h), 
                                fprime=lambda x: myFunPrime(x,omega[j], g,h), x0=kGuess, 
                                method='newton').root;
        kGuess = k[j]
        
    return k

def lcg(seed, a=1103515245, c=12345, m=2**31, n=1):
    # Linear congruential random number generator.
    # Chosen to have the sampe implementation in Matlab and Python.
    # See: https://en.wikipedia.org/wiki/Linear_congruential_generator
    
    numbers = np.ones(n)
    numbers *= seed
    for i in range(1, n):
        numbers[i] = (a * numbers[i-1] + c) % m
    return 2*np.pi*np.array([x / m for x in numbers])  # Normalize to [0, 1]

def generateRandomPhases(inputDict, seed=2):
    # Stating a seed will allow for repeatability
    # of your randomness
    # rng = np.random.default_rng(seed)
    # phi = 2*np.pi*rng.random(len(inputDict["Spectrum"]))
    phi = lcg(seed, n=len(inputDict["Spectrum"]))
        
    outputDict = dict()
    outputDict.update(inputDict)    
    outputDict["randomPhases"] = phi;
    
    return outputDict

def downsample(inputDict, dropEvery=2, listOfFields=None):
    
    # Check which fields we need to act onto
    if listOfFields is None:
        listOfFields = inputDict.keys() # all of them
    
    # Create the output dictionary
    outputDict = dict()
    outputDict.update(inputDict) # copy everything
    
    # Reduce the nr of samples in the desired field
    for field_ in listOfFields:
        outputDict[field_] = inputDict[field_][::dropEvery]

    # Increase the time step as well
    # If you drop one every two samples, the time step doubles
    outputDict["dt"] *= dropEvery    
    
    return outputDict

def pad2(vector, size):
    # Padding an array with zeros up to size
    return np.pad(vector, [1, size - len(vector) - 1]) 


class Timer(object):
    """ This is just a simple timer as we don't have tic - toc in Python.
    Taken from here: https://stackoverflow.com/questions/5849800/what-is-the-python-equivalent-of-matlabs-tic-and-toc-functions
    """
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        elapsedTime = time.time() - self.tstart
        print('Elapsed: %.5f' % (elapsedTime))
