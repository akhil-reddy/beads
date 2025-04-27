import numpy as np
from scipy.signal import butter, filtfilt


def meddis_ribbon_synapse(ihc_out, fs, params='Zilany2014'):
    """
    Phenomenological IHC–AN synapse with power‑law adaptation.
    Zilany et al. (2009,2014) model provides realistic adaptation and spontaneous rates.
    """
    # Here we’d call into the cochlea Python package:
    # from cochlea.external import run_matlab_auditory_periphery
    # spike_trains = run_matlab_auditory_periphery(ihc_out, fs, anf_num=(100,0,0), cf=..., seed=0)
    # For a pure Python stub:
    adapted = ihc_out ** 1.3  # simple approximation of power-law adaptation
    return adapted


def ihc_transduction(channel, fs, cutoff=1000):
    """
    Inner‐hair‑cell mechano‐electrical transduction:
    1) Half-wave rectification
    2) Lowpass filtering (~1 kHz)
    """
    rect = np.maximum(channel, 0)
    b, a = butter(1, cutoff / (fs / 2))
    return filtfilt(b, a, rect)
