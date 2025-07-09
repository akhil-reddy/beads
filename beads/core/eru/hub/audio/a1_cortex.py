import numpy as np
from scipy.signal import convolve2d


# TODO: Improve this to elements from ERU design
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
        self.fs = fs
        self.num_freqs = num_freqs
        self.num_taps = num_taps
        # Gaussian spectral tuning
        freqs = np.arange(num_freqs)
        self.spectral = np.exp(-0.5 * ((freqs - center_freq_idx) / freq_bandwidth) ** 2)
        # Temporal Gabor window
        times = np.arange(num_taps) / fs
        delay = best_delay_idx / fs
        self.temporal = np.exp(-0.5 * ((times - delay) / temporal_sigma) ** 2) * np.cos(
            2 * np.pi * modulation_rate * (times - delay))
        self.strf = weight * np.outer(self.spectral, self.temporal)

    def apply(self, spectrogram):
        # 2D convolution and sum across frequency
        conv = convolve2d(spectrogram, self.strf, mode='valid')
        return conv.sum(axis=0)


class ShortTermSynapse:
    """
    Implements Tsodyks-Markram depression and facilitation dynamics.
    Based on models by Tsodyks et al. (1998).
    u: utilization, R: resources
    """

    def __init__(self, U=0.5, tau_rec=0.8, tau_fac=0.0, dt=0.001):
        self.U = U
        self.R = 1.0
        self.u = U
        self.tau_rec = tau_rec
        self.tau_fac = tau_fac
        self.dt = dt

    def step(self, spike_train):  # spike_train: binary array
        I = np.zeros_like(spike_train, float)
        for t, s in enumerate(spike_train):
            if s:
                self.u += (self.U - self.u) * (1 - np.exp(-self.dt / self.tau_fac)) if self.tau_fac > 0 else self.U
                I[t] = self.u * self.R
                self.R -= self.u * self.R
            # recover resources
            self.R += (1 - self.R) * (1 - np.exp(-self.dt / self.tau_rec))
        return I


# TODO: Improve this to elements from ERU design
class ConductanceLIFNeuron:
    """
    Conductance-based LIF with synaptic and adaptation currents, per Brette & Gerstner (2005).
    """

    def __init__(self,
                 dt=0.001,
                 C_m=200e-12,  # membrane capacitance (F)
                 g_L=10e-9,  # leak conductance (S)
                 E_L=-0.070,  # leak reversal (V)
                 V_th=-0.050,  # threshold (V)
                 V_reset=-0.065,  # reset potential (V)
                 t_ref=0.002,  # refractory (s)
                 E_e=0.0,  # excitatory reversal (V)
                 tau_e=0.005,  # excitatory synapse decay (s)
                 w_e=1e-9,  # synaptic weight scale
                 a=2e-9,  # subthreshold adaptation conductance (S)
                 tau_w=0.2  # adaptation time constant (s)
                 ):
        self.dt = dt
        self.C_m = C_m
        self.g_L = g_L
        self.E_L = E_L
        self.V = E_L
        self.V_th = V_th
        self.V_reset = V_reset
        self.t_ref = t_ref
        self.refrac = 0
        self.E_e = E_e
        self.tau_e = tau_e
        self.w_e = w_e
        self.ge = 0.0
        self.a = a
        self.w = 0.0  # adaptation variable
        self.tau_w = tau_w

    def step(self, spike_input):
        # update synaptic conductance
        self.ge += self.w_e * spike_input
        # conductance decay
        self.ge -= self.dt * (self.ge / self.tau_e)
        # refractory handling
        if self.refrac > 0:
            self.refrac -= self.dt
            self.V = self.V_reset
            return False
        # membrane potential update
        I_L = self.g_L * (self.E_L - self.V)
        I_e = self.ge * (self.E_e - self.V)
        I_adapt = self.w * (self.E_L - self.V)
        dV = (I_L + I_e + I_adapt) / self.C_m
        self.V += self.dt * dV
        # adaptation variable
        self.w += self.dt * (self.a * (self.V - self.E_L) - self.w) / self.tau_w
        # spike condition
        if self.V >= self.V_th:
            self.V = self.V_reset
            self.refrac = self.t_ref
            return True
        return False


class PrimaryAuditoryCortex:
    """
    Biophysically-inspired A1 network with STRFs, dynamic synapses & conductance LIF neurons.
    """

    def __init__(self,
                 fs,
                 freq_res,
                 num_neurons=100,
                 num_freqs=128,
                 num_taps=64):
        self.fs = fs
        self.dt = 1.0 / fs
        self.units = []
        for _ in range(num_neurons):
            # random STRF
            cf = np.random.randint(10, num_freqs - 10)
            bw = np.random.uniform(1.0, 5.0)
            bd = np.random.randint(5, num_taps - 5)
            ts = np.random.uniform(0.01, 0.05)
            mr = np.random.uniform(2.0, 20.0)
            strf = SpectroTemporalReceptiveField(num_freqs, num_taps, cf, bw, bd, ts, mr, fs, freq_res)
            syn = ShortTermSynapse(U=0.4, tau_rec=0.5, tau_fac=0.0, dt=self.dt)
            neuron = ConductanceLIFNeuron(dt=self.dt)
            self.units.append({'strf': strf, 'syn': syn, 'neu': neuron})

    def run(self, spectrogram):
        T = spectrogram.shape[1] - self.units[0]['strf'].num_taps + 1
        spikes_out = {i: [] for i in range(len(self.units))}
        # compute drives
        for i, unit in enumerate(self.units):
            drive = unit['strf'].apply(spectrogram)
            # binarize drive for the synapse step (simplified)
            input_spikes = drive > np.percentile(drive, 90)
            syn_current = unit['syn'].step(input_spikes)
            for t in range(T):
                if unit['neu'].step(syn_current[t]):
                    spikes_out[i].append(t * self.dt)
        return spikes_out
