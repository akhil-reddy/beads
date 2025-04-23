import numpy as np
from scipy.signal import lfilter, butter


class Pinna:
    """
    Models spectral shaping by the pinna via
    a simple notch+peak filterbank derived from HRTF data.
    """

    def __init__(self, fs, notch_freqs, peak_freqs, notch_depths, peak_gains):
        """
        Args:
          fs (int): Sampling rate (Hz)
          notch_freqs (list[float]): center freqs of spectral notches (Hz)
          peak_freqs (list[float]): center freqs of spectral peaks (Hz)
          notch_depths (list[float]): depths of notches (dB)
          peak_gains (list[float]): gains of peaks (dB)
        """
        self.fs = fs
        # design biquad notch and peak filters
        self.notches = [self._design_notch(f0, d) for f0, d in zip(notch_freqs, notch_depths)]
        self.peaks = [self._design_peak(f0, g) for f0, g in zip(peak_freqs, peak_gains)]

    def _design_notch(self, f0, depth, Q=5):
        # notch filter (bandstop) with given depth (dB)
        w0 = f0 / (self.fs / 2)
        b, a = butter(2, [w0 / (Q), w0 * Q], btype='bandstop')
        # apply depth by mixing with identity
        gain = 10 ** (-depth / 20)
        b = gain * np.array([1, 0, 0]) + (1 - gain) * b
        return b, a

    def _design_peak(self, f0, gain_db, Q=5):
        # peak filter (bandpass) with given gain (dB)
        w0 = f0 / (self.fs / 2)
        b, a = butter(2, [w0 / (Q), w0 * Q], btype='bandpass')
        gain = 10 ** (gain_db / 20)
        b = gain * b
        return b, a

    def process(self, x):
        """
        Apply all notches then all peaks in series.
        Args:
          x (np.ndarray): mono audio signal
        Returns:
          y (np.ndarray): spectrally shaped signal
        """
        y = x
        for b, a in self.notches:
            y = lfilter(b, a, y)
        for b, a in self.peaks:
            y = lfilter(b, a, y)
        return y


class EarCanal:
    """
    Models ear-canal resonance as a 2nd-order bandpass
    centered at the canal’s quarter-wave frequency (~3 kHz).
    """

    def __init__(self, fs, length_m=0.025, gain_db=12):
        """
        Args:
          fs (int): Sampling rate
          length_m (float): ear canal length (~0.02–0.03 m) :contentReference[oaicite:6]{index=6}
          gain_db (float): desired resonant boost in dB (~10–20 dB) :contentReference[oaicite:7]{index=7}
        """
        self.fs = fs
        # quarter-wave resonance f = c/(4·L)
        c = 343.0  # speed of sound (m/s)
        f0 = c / (4 * length_m)
        bw = f0 / 3  # approximate bandwidth :contentReference[oaicite:8]{index=8}
        # design bandpass
        low = (f0 - bw / 2) / (fs / 2)
        high = (f0 + bw / 2) / (fs / 2)
        b, a = butter(2, [low, high], btype='bandpass')
        # apply gain
        g = 10 ** (gain_db / 20)
        self.b, self.a = g * b, a

    def process(self, x):
        """Boost frequencies around f0 by ~gain_db."""
        return lfilter(self.b, self.a, x)


class OuterEar:
    """
    Combines Pinna + EarCanal to form the outer-ear transfer function.
    """

    def __init__(self, fs):
        self.fs = fs
        # default notch/peak parameters from HRTF studies :contentReference[oaicite:9]{index=9}
        notch_freqs = [6e3, 8e3]  # spectral notches (Hz)
        notch_depths = [10, 8]  # dB
        peak_freqs = [4e3, 12e3]  # spectral peaks (Hz)
        peak_gains = [6, 4]  # dB
        self.pinna = Pinna(fs, notch_freqs, peak_freqs, notch_depths, peak_gains)
        # ear canal resonance (~3 kHz, ~15 dB) :contentReference[oaicite:10]{index=10}
        self.canal = EarCanal(fs, length_m=0.025, gain_db=15)

    def process(self, x):
        """Apply pinna then canal filtering."""
        return self.canal.process(self.pinna.process(x))


def initialize_outer_ear(model, fs):
    """
    Initialize the outer-ear model within a larger system.

    Args:
      model: parent object (e.g., a Head or HearingSystem)
      fs (int): sampling rate

    Returns:
      OuterEar instance attached to model
    """
    outer = OuterEar(fs)
    # attach to model (mimicking Retina example)
    model.outer_ear = outer
    return outer
