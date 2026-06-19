from dataclasses import dataclass

import numpy as np
from typing import Any

from models import Model


@dataclass
class Wind(Model):
    TDur: Any = None
    fHighCut: Any = None
    I: Any = None
    V_10: Any = None
    l: Any = None
    t: Any = None
    Spectrum: Any = None
    amplitudeSpectrum: Any = None
    f: Any = None
    randomPhases: Any = None
    V_hub: Any = None
    windTimeSeries: Any = None

    def calculate_kaimal_spectrum(self):
        df = self.TDur ** -1
        f = np.arange(df, self.fHighCut, df)

        spectrum = (4 * self.I ** 2 * self.V_10 * self.l) / (1 + 6 * f * self.l / self.V_10) ** (5 / 3)
        amplitude_spectrum = np.sqrt(2 * spectrum * df)

        self.Spectrum = spectrum
        self.amplitudeSpectrum = amplitude_spectrum
        self.f = f
        return self

    def generate_random_phases(self, seed=2):
        numbers = np.ones(len(self.Spectrum))
        numbers *= seed
        for i in range(1, len(numbers)):
            numbers[i] = (1103515245 * numbers[i - 1] + 12345) % (2 ** 31)
        self.randomPhases = 2 * np.pi * np.array([value / (2 ** 31) for value in numbers])
        return self

    def calculate_time_series(self):
        t = self.t
        f = self.f
        wind_time_series = np.zeros_like(t)

        random_phases = self.randomPhases if hasattr(self, "randomPhases") else np.random.uniform(0, 2 * np.pi, size=len(f))

        for i_, ti in enumerate(t):
            wind_time_series[i_] = np.sum(self.amplitudeSpectrum * np.cos(2 * np.pi * f * ti + random_phases))

        self.V_hub = wind_time_series + self.V_10
        self.randomPhases = random_phases
        self.windTimeSeries = wind_time_series
        return self

    def calculate_time_series_fft(self):
        t = self.t
        f = self.f

        M = len(t)
        wind_time_series_kernel = np.zeros(M, dtype=complex)
        for i in range(len(f)):
            amplitude_spectrum = self.amplitudeSpectrum[i]
            random_phases = self.randomPhases[i]

            if i + 1 < M:
                wind_time_series_kernel[i + 1] = amplitude_spectrum * np.exp(1j * random_phases)

        wind_time_series = np.fft.ifft(wind_time_series_kernel, n=M).real * M
        self.V_hub = wind_time_series + self.V_10
        return self


def calculateKaimalSpectrum(windDict):
    return Wind.from_mapping(windDict).calculate_kaimal_spectrum()


def calculateWindTimeSeries(windDict):
    return Wind.from_mapping(windDict).calculate_time_series()


def calculateWindTimeSeriesFFT(windDict):
    return Wind.from_mapping(windDict).calculate_time_series_fft()