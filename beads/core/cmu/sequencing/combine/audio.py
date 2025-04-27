import numpy as np
from scipy.signal import lfilter


def design_gammatone(fs, cf, order=4, bandwidth=None):
    """
    Design a 4th‐order Gammatone filter for center freq cf.
    Based on Slaney’s auditory toolbox implementation
    """
    if bandwidth is None:
        # ERB-scale: ERB(cf)=24.7 + 0.108*cf
        erb = 24.7 + 0.108 * cf
        bandwidth = 1.019 * erb
    # Approximate poles (simplified)…
    # (Full implementation would compute pole locations and filter coefficients)
    b = [1.0]
    a = [1.0, -2 * np.exp(-np.pi * bandwidth / fs) * np.cos(2 * np.pi * cf / fs), np.exp(-2 * np.pi * bandwidth / fs)]
    return b, a


def drnl_filter(channel, fs, cf, gain=1.0):
    """
    Dual‑Resonance Nonlinear (DRNL) model: linear + nonlinear pathways.
    Follows Zhang et al. (2001) phenomenological tuning with compression
    """
    # Linear path: gentle bandpass
    linear = channel * gain
    # Nonlinear path: half‑wave rect + power‑law compression + bandpass
    rect = np.maximum(channel, 0)
    compressed = rect ** 0.3  # compressive exponent
    # Bandpass again (reuse gammatone)
    b, a = design_gammatone(fs, cf, bandwidth=1.019 * (24.7 + 0.108 * cf))
    nonlinear = lfilter(b, a, compressed)
    return linear + nonlinear


def apply_cochlear_amplifier(channels, fs, center_freqs):
    """
    Amplify and sharpen each channel, simulating OHC feedback
    """
    out = []
    for ch, cf in zip(channels, center_freqs):
        out.append(drnl_filter(ch, fs, cf))
    return np.stack(out, axis=0)
