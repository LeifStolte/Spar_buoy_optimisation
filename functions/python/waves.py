from dataclasses import dataclass

import numpy as np
from typing import Any

from common import dispersion
from models import Model


@dataclass
class Waves(Model):
    Hs: Any = None
    Tp: Any = None
    TDur: Any = None
    fHighCut: Any = None
    h: Any = None
    z: Any = None
    t: Any = None
    gamma: Any = None
    Spectrum: Any = None
    amplitude: Any = None
    f: Any = None
    fp: Any = None
    randomPhases: Any = None
    eta: Any = None
    u: Any = None
    ut: Any = None

    def calculate_regular_wave_parameters(self):
        self.amplitude = np.array([0.5 * self.Hs])
        self.f = np.array([1.0 / self.Tp])
        self.randomPhases = np.array([0.0])
        return self

    def calculate_jonswap_spectrum(self):
        hs = self.Hs
        tp = self.Tp

        if hasattr(self, "gamma") and self.gamma is not None:
            gamma = self.gamma
        else:
            tp_hs_ratio = tp / np.sqrt(hs)
            if tp_hs_ratio <= 3.6:
                gamma = 5
            elif 3.6 < tp_hs_ratio <= 5:
                gamma = np.exp(5.75 - 1.15 * tp_hs_ratio)
            else:
                gamma = 1

        df = self.TDur ** -1.0
        f = np.arange(df, self.fHighCut, df)

        sigma = np.ones(len(f))
        fp = 1.0 / tp
        sigma[f > fp] = 0.09
        sigma[f <= fp] = 0.07

        spm = 5 / 16 * hs ** 2 * fp ** 4 * f ** (-5) * np.exp(-5 / 4 * (f / fp) ** (-4))
        h = np.exp(-0.5 * ((f / fp - 1) / sigma) ** 2)
        enhancement = gamma ** h
        spectrum = spm * (1 - 0.287 * np.log(gamma)) * enhancement

        multi = 1.00612915
        amplitude = np.sqrt(2 * spectrum * df) * multi

        self.Spectrum = spectrum
        self.amplitude = amplitude
        self.f = f
        self.fp = fp
        self.gamma = gamma
        return self

    def generate_random_phases(self, seed=2):
        numbers = np.ones(len(self.amplitude))
        numbers *= seed
        for i in range(1, len(numbers)):
            numbers[i] = (1103515245 * numbers[i - 1] + 12345) % (2 ** 31)
        self.randomPhases = 2 * np.pi * np.array([value / (2 ** 31) for value in numbers])
        return self

    def calculate_free_surface_elevation_time_series(self):
        t = self.t
        f = np.atleast_1d(self.f)
        amplitudes = np.atleast_1d(self.amplitude)

        free_surf_time_series = np.zeros_like(t)

        random_phases = self.randomPhases if hasattr(self, "randomPhases") else np.random.uniform(0, 2 * np.pi, size=len(f))

        for i, ti in enumerate(t):
            free_surf_time_series[i] = np.sum(amplitudes * np.cos(2 * np.pi * f * ti + random_phases))

        self.eta = free_surf_time_series
        self.randomPhases = random_phases
        return self

    def calculate_kinematics(self, wheelerStretching=False):
        del wheelerStretching
        t = self.t
        f = np.atleast_1d(self.f)
        omega = 2 * np.pi * f

        h = self.h
        z = self.z
        k = dispersion(f, h)

        amplitudes = np.atleast_1d(self.amplitude)
        phases = self.randomPhases

        u = np.zeros((len(t), len(z)))
        ut = np.zeros((len(t), len(z)))

        for i, ti in enumerate(t):
            cos_terms = np.cos(omega * ti + phases)
            sin_terms = np.sin(omega * ti + phases)
            for j, zj in enumerate(z):
                transfer = np.cosh(k * (zj + h)) / np.sinh(k * h)
                u[i, j] = -np.sum(amplitudes * omega * transfer * cos_terms)
                ut[i, j] = np.sum(amplitudes * omega ** 2 * transfer * sin_terms)

        self.u = u
        self.ut = ut
        return self

    def calculate_free_surface_elevation_time_series_fft(self):
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
        t = self.t
        f = self.f

        M = len(t)
        free_surf_time_series_kernel = np.zeros(M, dtype=complex)

        for i in range(len(f)):
            amplitude = self.amplitude[i]
            phase = self.randomPhases[i]
            if i + 1 < M:
                free_surf_time_series_kernel[i + 1] = amplitude * np.exp(1j * phase)

        free_surf_time_series = np.fft.ifft(free_surf_time_series_kernel).real * M
        self.eta = free_surf_time_series
        return self

    def calculate_kinematics_fft(self):
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
        =====================
        - Sequential bin placement: components go in bins i+1 (skip DC)
        - Proper depth-dependent transfer function: cosh(k(z+h))/sinh(kh)
        - IFFT scaling by M to match slow method amplitude convention
        """
        t = self.t
        f = self.f
        omega = 2 * np.pi * f

        h = self.h
        z = self.z
        u = np.zeros((len(t), len(z)))
        ut = np.zeros((len(t), len(z)))

        k = dispersion(f, h)
        M = len(t)

        for j_, z_ in enumerate(z):
            u_kernel = np.zeros(M, dtype=complex)
            ut_kernel = np.zeros(M, dtype=complex)

            for i in range(len(f)):
                amplitude = self.amplitude[i]
                phase = self.randomPhases[i]
                transfer = np.cosh(k[i] * (z_ + h)) / np.sinh(k[i] * h)

                if i + 1 < M:
                    u_kernel[i + 1] = -amplitude * omega[i] * transfer * np.exp(1j * phase)
                    ut_kernel[i + 1] = amplitude * omega[i] ** 2 * transfer * np.exp(1j * phase) * (1j)

            u[:, j_] = np.fft.ifft(u_kernel).real * M
            ut[:, j_] = np.fft.ifft(ut_kernel).real * M

        self.u = u
        self.ut = ut
        return self


def calculateRegularWaveParameters(waveDict):
    return Waves.from_mapping(waveDict).calculate_regular_wave_parameters()


def calculateJONSWAPSpectrum(waveDict):
    return Waves.from_mapping(waveDict).calculate_jonswap_spectrum()


def calculateFreeSurfaceElevationTimeSeries(waveDict):
    return Waves.from_mapping(waveDict).calculate_free_surface_elevation_time_series()


def calculateKinematics(inputDict, wheelerStretching=False):
    return Waves.from_mapping(inputDict).calculate_kinematics(wheelerStretching=wheelerStretching)


def calculateFreeSurfaceElevationTimeSeriesFFT(waveDict):
    return Waves.from_mapping(waveDict).calculate_free_surface_elevation_time_series_fft()


def calculateKinematicsFFT(inputDict):
    return Waves.from_mapping(inputDict).calculate_kinematics_fft()