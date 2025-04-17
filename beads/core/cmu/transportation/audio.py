import numpy as np


def lif_spike_generator(ihc_signal, fs, threshold=0.1, tau=0.001):
    """
    Leaky integrate‑and‑fire model of AN fibers:
    dV/dt = -V/tau + I(t)/C + noise
    with spike at V>threshold → V reset.
    Based on Brian Hears example.
    """
    dt = 1 / fs
    V = 0.0
    spikes = []
    for t, I in enumerate(ihc_signal):
        dV = (-V / tau + I) * dt
        V += dV + np.sqrt(dt) * 0.01 * np.random.randn()
        if V >= threshold:
            spikes.append(t * dt)
            V = 0.0
    return np.array(spikes)


def generate_spike_trains(adapted, fs, center_freqs):
    """
    Convert each channel’s adapted signal into a train of spike times.
    """
    all_trains = []
    for ch in adapted:
        trains = lif_spike_generator(ch, fs)
        all_trains.append(trains)
    return all_trains
