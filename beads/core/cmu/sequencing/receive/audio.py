import numpy as np
from scipy.signal import lfilter, butter


class Pinna:
    """
    Spectral shaping by the pinna via notch + peak filters from HRTF data.
    """

    def __init__(self, fs, notch_freqs, notch_depths, peak_freqs, peak_gains, Q=5):
        self.fs = fs
        self.notches = [self._design_notch(f0, depth, Q) for f0, depth in zip(notch_freqs, notch_depths)]
        self.peaks = [self._design_peak(f0, gain_db, Q) for f0, gain_db in zip(peak_freqs, peak_gains)]

    def _design_notch(self, f0, depth, Q):
        # Bandstop around f0 with depth (dB)
        b, a = butter(2, [f0 / np.sqrt(Q) / (self.fs / 2), f0 * np.sqrt(Q) / (self.fs / 2)],
                      btype='bandstop', fs=self.fs)
        gain = 10 ** (-depth / 20)
        # Mix with identity to apply depth
        b = gain * np.array([1, 0, 0]) + (1 - gain) * b
        return b, a

    def _design_peak(self, f0, gain_db, Q):
        # Bandpass around f0 with gain (dB)
        b, a = butter(2, [f0 / np.sqrt(Q) / (self.fs / 2), f0 * np.sqrt(Q) / (self.fs / 2)],
                      btype='bandpass', fs=self.fs)
        b *= 10 ** (gain_db / 20)
        return b, a

    def process(self, x):
        y = x
        for b, a in self.notches:
            y = lfilter(b, a, y)
        for b, a in self.peaks:
            y = lfilter(b, a, y)
        return y


class EarCanal:
    """
    Ear-canal resonance as 2nd-order bandpass at ~c/(4L).
    """

    def __init__(self, fs, length_m=0.025, gain_db=15):
        self.fs = fs
        c = 343.0
        f0 = c / (4 * length_m)
        bw = f0 / 3
        low, high = f0 - bw / 2, f0 + bw / 2
        b, a = butter(2, [low / (fs / 2), high / (fs / 2)], btype='bandpass', fs=self.fs)
        b *= 10 ** (gain_db / 20)
        self.b, self.a = b, a

    def process(self, x):
        return lfilter(self.b, self.a, x)


class OuterEar:
    def __init__(self, fs):
        notch_freqs = [6000, 8000]
        notch_depths = [10, 8]
        peak_freqs = [4000, 12000]
        peak_gains = [6, 4]
        self.pinna = Pinna(fs, notch_freqs, notch_depths, peak_freqs, peak_gains)
        self.canal = EarCanal(fs)

    def process(self, x):
        return self.canal.process(self.pinna.process(x))


def initialize_outer_ear(model, fs):
    model.outer_ear = OuterEar(fs)
    return model.outer_ear
