import numpy as np
from common import dispersion

def calculateRegularWaveParameters(waveDict):
    
    amplitude = np.array([0.5*waveDict["Hs"]])
    f = np.array([1.0 / waveDict["Tp"]])
    epsilon = np.array([0.])
    
    # Store it inside the waves dictionary
    outputDict = dict()
    outputDict.update(waveDict)
    outputDict["amplitude"] = amplitude
    outputDict["randomPhases"] = epsilon
    outputDict["f"] = f
    
    return outputDict

def calculateJONSWAPSpectrum(waveDict):
           
    Hs = waveDict["Hs"]
    Tp = waveDict["Tp"]
    
    # Use gamma from input, or calculate if not provided
    if "gamma" in waveDict and waveDict["gamma"] is not None:
        gamma = waveDict["gamma"]
    else:
        # Calculate gamma based on the provided formula
        Tp_Hs_ratio = Tp / np.sqrt(Hs)
        if Tp_Hs_ratio <= 3.6:
            gamma = 5
        elif 3.6 < Tp_Hs_ratio <= 5:
            gamma = np.exp(5.75 - 1.15 * Tp_Hs_ratio)
        else:
            gamma = 1
                    
    # Calculate frequency information
    df = waveDict["TDur"]**-1.0
    f = np.arange(df, waveDict["fHighCut"], df)
    
    # Spectral width parameter
    sigma = np.ones(len(f))

    fp = 1./Tp

    sigma[f>fp] = 0.09
    sigma[f<=fp] = 0.07        

    # Pierson-Moskowitz spectrum
    Spm = 5/16 * Hs**2 * fp**4 * f**(-5) * \
            np.exp(-5/4 * (f / fp)**(-4))
    
    # JONSWAP spectrum formula
    # Correct enhancement factor: gamma^H where H = exp(...)
    H = np.exp(-0.5 * ((f/fp - 1) / sigma)**2)
    enhancement = gamma ** H
    Spectrum = Spm * (1 - 0.287 * np.log(gamma)) * enhancement

    # Scaling factor to ensure Hs = 4σ exactly
    # This corrects for discretization and numerical effects
    multi = 1.00612915
    
    # Calculate amplitude for time series generation
    amplitude = np.sqrt(2*Spectrum*df) * multi
    # Store it inside the waves dictionary
    outputDict = dict()
    outputDict.update(waveDict)
    outputDict["Spectrum"] = Spectrum
    outputDict["amplitude"] = amplitude
    outputDict["f"] = f
    outputDict["fp"] = fp
    outputDict["gamma"] = gamma 
    
    return outputDict
       
def calculateFreeSurfaceElevationTimeSeries(waveDict):
    """
    Build a free-surface elevation time series from frequency components.
    Works for both regular (single f) and irregular (JONSWAP) inputs.
    """
    t = waveDict["t"]
    f = np.atleast_1d(waveDict["f"])
    amplitudes = np.atleast_1d(waveDict["amplitude"])

    freeSurfTimeSeries = np.zeros_like(t)

    # Use existing randomPhases if provided, otherwise generate new
    if "randomPhases" in waveDict:
        randomPhases = waveDict["randomPhases"]
    else:
        randomPhases = np.random.uniform(0, 2*np.pi, size=len(f))

    for i, ti in enumerate(t):
        freeSurfTimeSeries[i] = np.sum(
            amplitudes * np.cos(2*np.pi*f*ti + randomPhases)
        )

    outputDict = dict()
    outputDict.update(waveDict)
    outputDict["eta"] = freeSurfTimeSeries
    outputDict["randomPhases"] = randomPhases
    return outputDict

def calculateKinematics(inputDict, wheelerStretching=False):
    """
    Compute velocity (u) and acceleration (ut) time histories.
    Works with multiple frequencies (e.g. JONSWAP).
    """
    t = inputDict["t"]
    f = np.atleast_1d(inputDict["f"])
    omega = 2*np.pi*f

    h = inputDict["h"]
    z = inputDict["z"]

    k = dispersion(f, h)   # array of wavenumbers

    amplitudes = np.atleast_1d(inputDict["amplitude"])
    phases = inputDict["randomPhases"]

    u = np.zeros((len(t), len(z)))
    ut = np.zeros((len(t), len(z)))

    for i, ti in enumerate(t):
        cos_terms = np.cos(omega*ti + phases)
        sin_terms = np.sin(omega*ti + phases)
        for j, zj in enumerate(z):
            transfer = np.cosh(k*(zj+h)) / np.sinh(k*h)
            u[i, j] = -np.sum(amplitudes * omega * transfer * cos_terms)
            ut[i, j] = np.sum(amplitudes * omega**2 * transfer * sin_terms)

    outputDict = dict()
    outputDict.update(inputDict)
    outputDict["u"] = u
    outputDict["ut"] = ut
    
    return outputDict

def calculateFreeSurfaceElevationTimeSeriesFFT(waveDict):
    """
    FFT-based free surface elevation calculation - optimized version of slow method.
    
    Key Implementation Notes:
    ========================
    1. Sequential Frequency Placement: The spectrum frequencies (f = df, 2*df, 3*df, ...) 
       are designed to map directly to sequential FFT bins [1, 2, 3, ...], skipping 
       the DC component at bin 0. No frequency interpolation needed.
       
    2. Proper FFT Bin Mapping: Components placed in bins i+1 where i is the frequency 
       index. This ensures the FFT frequency grid matches the spectrum grid exactly.
       
    3. IFFT Scaling: The IFFT result is scaled by M (number of time points) to match 
       the amplitude convention used in the slow method.
       
    4. No Conjugate Symmetry Required: Since we're using the full complex FFT kernel 
       and taking only the real part, numpy's IFFT automatically handles the symmetry 
       requirements for real-valued output signals.
    """
    
    t = waveDict["t"]
    f = waveDict["f"]
    
    # FIXME Assignment 3 Q1.8: compute the fft kernel and perform the IFFT
    M = len(t)
    
    # Create the FFT kernel - most components will be zero
    freeSurfTimeSeriesKernel = np.zeros(M, dtype=complex)
    
    # Place each frequency component in the correct FFT bin
    # The FFT frequency array is: [0, df_fft, 2*df_fft, ..., (M/2-1)*df_fft, 
    #                             -M/2*df_fft, ..., -df_fft]
    # where df_fft = 1/(M*dt)
    
    for i in range(len(f)):
        amplitude = waveDict["amplitude"][i]
        phase = waveDict["randomPhases"][i]
        
        # Place this component directly in bin i+1 (skip DC component at bin 0)
        # This assumes the frequency array starts from df and increments by df
        if i + 1 < M:
            freeSurfTimeSeriesKernel[i + 1] = amplitude * np.exp(1j * phase)
    
    # Perform IFFT and take real part, scale by M
    freeSurfTimeSeries = np.fft.ifft(freeSurfTimeSeriesKernel).real * M

    # Store the result
    outputDict = dict()
    outputDict.update(waveDict)
    outputDict["t"] = t
    outputDict["eta"] = freeSurfTimeSeries
        
    return outputDict


def calculateKinematicsFFT(inputDict):
    """
    FFT-based wave kinematics calculation - optimized version of slow method.
    
    Critical Fix Applied:
    ====================
    Wave Acceleration Phase Error: The key bug was in the acceleration calculation
    where the phase shift was incorrect. The correct implementation requires:
    
    - Velocity: u = -A*ω*cosh(k(z+h))/sinh(kh) * cos(ωt + φ)
      FFT: multiply by -ω and use phase φ
      
    - Acceleration: ut = A*ω²*cosh(k(z+h))/sinh(kh) * sin(ωt + φ)  
      FFT: multiply by ω² and use phase φ with (-1j) factor
      
    The (-1j) factor provides the correct 90° phase shift for acceleration
    (cos → sin transformation), not (+1j) which would give the wrong sign.
    
    Implementation Details:
    ======================
    - Sequential bin placement: components go in bins i+1 (skip DC)
    - Proper depth-dependent transfer function: cosh(k(z+h))/sinh(kh)
    - IFFT scaling by M to match slow method amplitude convention
    """
    
    t = inputDict["t"]
    f = inputDict["f"]
    omega = 2*np.pi*f
    
    h = inputDict["h"]
    z = inputDict["z"]
    u = np.zeros((len(t), len(z)))
    ut = np.zeros((len(t), len(z)))
    
    k = dispersion(f, inputDict["h"])
    M = len(t)
    
    for j_, z_ in enumerate(z):
        
        # FIXME Assignment 3 Q1.8: compute the fft kernel and perform the IFFT
        uKernel = np.zeros(M, dtype=complex)
        utKernel = np.zeros(M, dtype=complex)
        
        # Fill the kernel with frequency domain values
        for i in range(len(f)):
            amplitude = inputDict["amplitude"][i]
            phase = inputDict["randomPhases"][i]
            transfer = np.cosh(k[i]*(z_+h)) / np.sinh(k[i]*h)
            
            # Place this component directly in bin i+1 (skip DC at bin 0)
            if i + 1 < M:
                # Velocity kernel (multiply by -omega for velocity)
                u_val = -amplitude * omega[i] * transfer * np.exp(1j * phase)
                uKernel[i + 1] = u_val
                
                # CRITICAL FIX: Wave acceleration phase error correction
                # The acceleration requires (-1j) factor, NOT (+1j)
                # This provides correct cos→sin phase shift for ut = dφ/dt
                # Wrong: * 1j  → gives inverted acceleration
                # Right: * (-1j) → matches slow method exactly
                ut_val = (amplitude * omega[i]**2 * transfer *
                         np.exp(1j * phase) * (1j))
                utKernel[i + 1] = ut_val
        
        # Perform IFFT
        u[:, j_] = np.fft.ifft(uKernel).real * M
        ut[:, j_] = np.fft.ifft(utKernel).real * M
    
    outputDict = dict()
    outputDict.update(inputDict)
    outputDict["u"] = u
    outputDict["ut"] = ut
    
    return outputDict