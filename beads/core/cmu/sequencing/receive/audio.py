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


def bank_of_gammatones(x, fs, center_freqs):
    """
    Apply a gammatone filterbank to signal x.
    Each channel approximates a place on the basilar membrane
    """
    channels = []
    for cf in center_freqs:
        b, a = design_gammatone(fs, cf)
        y = lfilter(b, a, x)
        channels.append(y)
    return np.stack(channels, axis=0)
