import numpy as np
from scipy.signal import convolve2d


class SpectroTemporalReceptiveField:
    """
    Implements 2D Gabor-like STRFs for A1 neurons (frequency × time).
    Based on Depireux et al. (2001) models of cat A1.
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
                 freq_res,
                 weight=1.0):
        self.num_freqs = num_freqs
        self.num_taps = num_taps
        self.fs = fs
        self.freq_res = freq_res
        self.weight = weight
        # build Gaussian spectral profile
        freqs = np.arange(num_freqs)
        self.spectral = np.exp(-0.5 * ((freqs - center_freq_idx) / freq_bandwidth) ** 2)
        # build temporal Gabor (cosine-modulated Gaussian)
        times = np.arange(num_taps) / fs
        delay = best_delay_idx / fs
        self.temporal = np.exp(-0.5 * ((times - delay) / temporal_sigma) ** 2) * \
                        np.cos(2 * np.pi * modulation_rate * (times - delay))
        # outer product yields STRF
        self.strf = weight * np.outer(self.spectral, self.temporal)

    def apply(self, spectrogram):
        """
        Convolve STRF with input spectrogram (freq × time) and return response over time.
        spectrogram: 2D array shape (freq × time)
        returns: 1D array of neural drive (time)
        """
        # full 2D convolution, valid mode yields (freq-f+1, time-t+1)
        conv = convolve2d(spectrogram, self.strf, mode='valid')
        # sum across frequency rows to get time series
        drive = conv.sum(axis=0)
        return drive


class A1Neuron:
    """
    Simple linear-nonlinear Poisson neuron for A1.
    Drive → spike rate via exp nonlinearity → Poisson spiking.
    """

    def __init__(self,
                 strf: SpectroTemporalReceptiveField,
                 baseline_rate=5.0,  # spikes/s
                 gain=1.0,
                 dt=0.001):  # time bin (s)
        self.strf = strf
        self.baseline = baseline_rate
        self.gain = gain
        self.dt = dt

    def simulate(self, spectrogram):
        """
        Given a spectrogram (freq × time), produce spike times (s).
        """
        drive = self.strf.apply(spectrogram)
        # instantaneous rate (spikes/s)
        rate = self.baseline + self.gain * np.exp(drive)
        # generate Poisson spikes per bin
        spike_bins = np.where(np.random.rand(len(rate)) < rate * self.dt)[0]
        return spike_bins * self.dt


class PrimaryAuditoryCortex:
    """
    Network of A1 neurons with diverse STRFs.
    Inputs: spectrogram (freq × time)
    Outputs: list of spike trains (s) per neuron.
    """

    def __init__(self,
                 fs,
                 freq_res,
                 num_neurons=100,
                 num_freqs=128,
                 num_taps=64):
        self.fs = fs
        self.dt = 1.0 / fs
        self.freq_res = freq_res
        self.num_neurons = num_neurons
        # Create a diverse bank of STRFs
        self.neurons = []
        for _ in range(num_neurons):
            # randomize STRF parameters within physiological ranges
            cf = np.random.randint(10, num_freqs - 10)
            bw = np.random.uniform(1.0, 5.0)
            bd = np.random.randint(5, num_taps - 5)
            ts = np.random.uniform(0.01, 0.05)  # 10–50 ms
            mr = np.random.uniform(2.0, 20.0)  # 2–20 Hz modulation
            strf = SpectroTemporalReceptiveField(
                num_freqs, num_taps, cf, bw, bd, ts, mr, fs, freq_res)
            neuron = A1Neuron(strf, dt=self.dt)
            self.neurons.append(neuron)

    def run(self, spectrogram):
        """
        Run all A1 neurons on the input spectrogram.
        Returns dict of spike trains per neuron index.
        """
        outputs = {}
        for idx, neuron in enumerate(self.neurons):
            spikes = neuron.simulate(spectrogram)
            outputs[idx] = spikes
        return outputs
