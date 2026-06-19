import pylab as plt
import numpy as np

from models import LoadSeries, Response, Structure, TimeInfo

def makeplots(wind, waves, structure, response, timeInfo, color, **kwargs):
    if not hasattr(wind, "copy_with"):
        wind = type("WindLike", (), wind)() if isinstance(wind, dict) else wind
    if not hasattr(waves, "copy_with"):
        waves = type("WavesLike", (), waves)() if isinstance(waves, dict) else waves
    if not hasattr(response, "copy_with"):
        response = Response.from_mapping(response)
    if not hasattr(structure, "copy_with"):
        structure = Structure.from_mapping(structure)
    if not hasattr(timeInfo, "copy_with"):
        timeInfo = TimeInfo.from_mapping(timeInfo)

    # Check if the axis has been passed otherwise initialize plot
    ax = kwargs.get("ax", np.zeros(2))
    if not ax.any():
        _, ax = plt.subplots(4,2, sharex='col')
    
    mask = response.t > timeInfo.TTrans

    ax[0,0].plot(wind.t, wind.V_hub, color=color)
    ax[0,0].set_ylabel("V[m/s]")
    
    f, _, S = freqSpectrum(wind.t, wind.V_hub)
    ax[0,1].plot(f, S, color=color)
    ax[0,1].set_ylabel("PSD [m^2/s^2 / Hz]")
    
    ax[1,0].plot(waves.t, waves.eta, color=color)
    ax[1,0].set_ylabel("eta[m]")
    
    f, _, S = freqSpectrum(waves.t, waves.eta)
    ax[1,1].plot(f, S, color=color)
    ax[1,1].set_ylabel("PSD [m^2 / Hz]")    

    ax[2,0].plot(response.t, response.x1, color=color)
    ax[2,0].set_ylabel("surge[m]")
    
    f, _, S = freqSpectrum(response.t[mask], response.x1[mask])
    ax[2,1].plot(f, S, color=color)
    ax[2,1].set_ylabel("PSD [m^2 / Hz]")    
    ax[2,1].axvline(structure.fnat[0], 0., np.max(S), color='k', linestyle='--',label='_nolegend_')

    ax[3,0].plot(response.t, np.rad2deg(response.x5), color=color)
    ax[3,0].set_ylabel("pitch[rad]")
    
    f, _, S = freqSpectrum(response.t[mask], response.x5[mask])
    ax[3,1].plot(f, S, color=color)
    ax[3,1].set_ylabel("PSD [rad^2 / Hz]")
    ax[3,1].axvline(structure.fnat[1], 0., np.max(S), color='k', linestyle='--',label='_nolegend_')
    
    ax[3,1].set_xlim([0., timeInfo.fHighCut])
    for ax_ in ax.ravel():
        ax_.grid(True)
    
    plt.tight_layout()
    
    return ax

    

def makeplotsMonopile(wind, waves, loads, response, timeInfo, color, **kwargs):
    if not hasattr(wind, "copy_with"):
        wind = type("WindLike", (), wind)() if isinstance(wind, dict) else wind
    if not hasattr(waves, "copy_with"):
        waves = type("WavesLike", (), waves)() if isinstance(waves, dict) else waves
    if not hasattr(loads, "copy_with"):
        loads = LoadSeries.from_mapping(loads)
    if not hasattr(response, "copy_with"):
        response = Response.from_mapping(response)
    if not hasattr(timeInfo, "copy_with"):
        timeInfo = TimeInfo.from_mapping(timeInfo)

    # Check if the axis has been passed otherwise initialize plot
    ax = kwargs.pop("ax", np.zeros(2))
    if not ax.any():
        _, ax = plt.subplots(4,2, sharex='col')
    
    ax[0,0].plot(wind.t, wind.V_hub, color=color, **kwargs)
    ax[0,0].set_ylabel("V[m/s]")
    
    f, _, S = freqSpectrum(wind.t, wind.V_hub)
    ax[0,1].plot(f, S, color=color, **kwargs)
    ax[0,1].set_ylabel("PSD [m^2/s^2 / Hz]")
    
    ax[1,0].plot(waves.t, waves.eta, color=color, **kwargs)
    ax[1,0].set_ylabel("eta[m]")
    
    f, _, S = freqSpectrum(waves.t, waves.eta)
    ax[1,1].plot(f, S, color=color, **kwargs)
    ax[1,1].set_ylabel("PSD [m^2 / Hz]")

    ax[2,0].plot(response.t, response.alpha, color=color, **kwargs)
    ax[2,0].set_ylabel("alpha[m]")
    
    f, _, S = freqSpectrum(response.t, response.alpha)
    ax[2,1].plot(f, S, color=color, **kwargs)
    ax[2,1].set_ylabel("PSD [m^2 / Hz]")    

    ax[3,0].plot(loads.t, np.rad2deg(loads.M), color=color, **kwargs)
    ax[3,0].set_ylabel("moment[Nm]")
    
    f, _, S = freqSpectrum(loads.t, loads.M)
    ax[3,1].plot(f, S, color=color, **kwargs)
    ax[3,1].set_ylabel("PSD [(Nm)^2 / Hz]")
        
    ax[3,1].set_xlim([0., timeInfo.fHighCut])
    for ax_ in ax.ravel():
        ax_.grid(True)
    
    plt.tight_layout()
    
    return ax

def freqSpectrum(t, x, flagMean=False):
    
    # Takes a time vector and a signal and returns the one-sided Fourier 
    # coefficients obtained with FFT, as well as the PSD function

    # flagmean=False: f[0]=df, a[0]=mean(x) is removed
    # flagmean=True: f[0]=0,  a[0]=mean(x) is kept
    
    if len(t) == 1:
        raise ValueError("The signal has only one element.")
    
    else:
        Tmax = t[-1] - t[0]
        df = Tmax**-1
        
        if flagMean:
            f = np.arange(len(t))*df
        else:
            f = (np.arange(1,len(t)+1)*df)
            
        a = np.fft.fft(x) / len(t)
        
        dt = t[1] - t[0]        
        f_nyq = 1./2./dt        
        a[f > f_nyq] = 0.
        
        if flagMean:
            a[1:] = 2*a[1:]
        else:
            a = 2*a[1:len(f)]
            a = np.append(a, 0.)
            
        # PSD function
        
        S = np.abs(a)**2 / (2*df)
        
        return f, a, S
        
        
