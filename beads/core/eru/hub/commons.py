# ----------------------------------------------------------------------------
# Spectro-Temporal Receptive Field (STRF) based on ERU/Gabor model enhancements
# ----------------------------------------------------------------------------
import numpy as np
from scipy.signal import fftconvolve


class SpectroTemporalReceptiveField:
    """
    Implements 2D Gabor-like STRFs for A1 neurons (frequency × time).
    Enhanced with normalization and separable convolutions.
    """

    def __init__(self,
                 num_freqs,
                 num_taps,
                 center_freq_idx,
                 freq_bandwidth,
                 best_delay_idx,
                 temporal_sigma,
                 modulation_rate,
                 fs,
                 weight=1.0):
        self.fs = fs
        self.num_freqs = num_freqs
        self.num_taps = num_taps
        # Spectral profile: Gaussian
        freqs = np.arange(num_freqs)
        self.spectral = np.exp(-0.5 * ((freqs - center_freq_idx) / freq_bandwidth) ** 2)
        # Normalize spectral kernel
        self.spectral /= np.linalg.norm(self.spectral)
        # Temporal Gabor kernel
        times = np.arange(num_taps) / fs
        delay = best_delay_idx / fs
        gauss = np.exp(-0.5 * ((times - delay) / temporal_sigma) ** 2)
        carrier = np.cos(2 * np.pi * modulation_rate * (times - delay))
        self.temporal = gauss * carrier
        # Normalize temporal kernel
        self.temporal /= np.linalg.norm(self.temporal)
        # Combined STRF weight
        self.strf = weight * np.outer(self.spectral, self.temporal)

    def apply(self, spectrogram):
        # Use FFT-based convolution for speed
        conv = fftconvolve(spectrogram, self.strf, mode='valid')
        # Sum over frequency axis to get temporal drive
        return np.sum(conv, axis=0)