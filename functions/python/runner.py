import numpy as np

from common import downsample, generateRandomPhases
from integration import dqdt, ode4
from loads import calculateDynamicLoads, calculateStaticWaveLoads, calculateStaticWindLoads
from models import LoadSeries, Response, Structure, TimeInfo
from monopile import computeElementwiseQuantities
from rotor import Rotor
from waves import Waves
from wind import Wind


def runEnvironmentalCondition(wind, waves, rotor, monopile, timeInfo):
    if not hasattr(wind, "copy_with"):
        wind = Wind.from_mapping(wind)
    if not hasattr(waves, "copy_with"):
        waves = Waves.from_mapping(waves)
    if not hasattr(rotor, "copy_with"):
        rotor = Rotor.from_mapping(rotor)
    if not hasattr(monopile, "copy_with"):
        monopile = Structure.from_mapping(monopile)
    if not hasattr(timeInfo, "copy_with"):
        timeInfo = TimeInfo.from_mapping(timeInfo)

    waves = waves.copy_with(TDur=timeInfo.TDur, fHighCut=timeInfo.fHighCut)
    wind = wind.copy_with(TDur=timeInfo.TDur, fHighCut=timeInfo.fHighCut)
    waves.t = np.arange(0.0, timeInfo.TDur, timeInfo.dt)
    wind.t = np.arange(0.0, timeInfo.TDur, timeInfo.dt)

    wind = wind.calculate_kaimal_spectrum()
    wind = wind.generate_random_phases(seed=wind.randomSeed)
    wind = wind.calculate_time_series_fft()

    waves = waves.calculate_jonswap_spectrum()
    waves = generateRandomPhases(waves, seed=waves.randomSeed)
    waves = waves.calculate_free_surface_elevation_time_series_fft()
    waves = waves.calculate_kinematics_fft()

    q0 = [0.0, 0.0]
    t_integration = np.arange(0.0, timeInfo.TDur, 2 * timeInfo.dt)
    q = ode4(dqdt, t_integration, q0, monopile, rotor, waves, wind)

    response = Response(
        t=t_integration,
        alphaDot=q[:, 1],
        alphaDotDot=np.gradient(q[:, 1], t_integration),
    )

    wind_downsampled = downsample(wind, dropEvery=2, listOfFields=["t", "V_hub"])
    wind_loads = calculateStaticWindLoads(wind_downsampled, rotor, monopile, response)

    waves_downsampled = downsample(waves, dropEvery=2, listOfFields=["t", "u", "ut", "eta"])
    wave_loads = calculateStaticWaveLoads(waves_downsampled, monopile, response)

    monopile = computeElementwiseQuantities(monopile)
    dynamic_loads = calculateDynamicLoads(monopile, response)

    output = {
        "waves": wave_loads,
        "wind": wind_loads,
        "dynamic": dynamic_loads,
    }
    output["total"] = LoadSeries(
        t=wave_loads.t.copy(),
        F=wave_loads.F + wind_loads.F + dynamic_loads.F,
        M=wave_loads.M + wind_loads.M + dynamic_loads.M,
    )

    return output