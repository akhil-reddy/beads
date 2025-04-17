import numpy as np
from scipy.signal import lfilter

from beads.core.cmu.sequencing.receive.audio import design_gammatone


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
