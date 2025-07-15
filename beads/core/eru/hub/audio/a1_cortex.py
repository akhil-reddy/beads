import numpy as np
from scipy.signal import fftconvolve


# ----------------------------------------------------------------------------
# Spectro-Temporal Receptive Field (STRF) based on ERU/Gabor model enhancements
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Tsodyks-Markram dynamic synapse with vectorized exponential updates
# ----------------------------------------------------------------------------
class ShortTermSynapse:
    """
    Tsodyks-Markram model: depression & facilitation with exact updates.
    """

    def __init__(self, U=0.5, tau_rec=0.8, tau_fac=0.0, dt=0.001):
        self.U = U
        self.tau_rec = tau_rec
        self.tau_fac = tau_fac
        self.dt = dt
        # Precompute exponential decay factors
        self.e_rec = np.exp(-dt / tau_rec)
        self.e_fac = np.exp(-dt / tau_fac) if tau_fac > 0 else 0.
        self.R = 1.0
        self.u = U

    def step(self, spike_train):
        # Vectorized pre-allocation
        T = spike_train.shape[0]
        I = np.zeros(T, dtype=float)
        for t in range(T):
            if spike_train[t]:
                # Facilitation dynamics (exact)
                if self.tau_fac > 0:
                    self.u = self.u * self.e_fac + self.U * (1 - self.e_fac)
                else:
                    self.u = self.U
                # Compute release
                I[t] = self.u * self.R
                self.R -= I[t]
            # Recovery (exact)
            self.R = 1 - (1 - self.R) * self.e_rec
        return I


# TODO: Bring in elements from ERU Design
# ----------------------------------------------------------------------------
# Conductance-based LIF neuron with adaptation and inhibitory channel
# ----------------------------------------------------------------------------
class ConductanceLIFNeuron:
    """
    Conductance-based LIF with synaptic excitation, inhibition & spike-frequency adaptation.
    Based on Brette & Gerstner (2005), with added inhibitory channel.
    """

    def __init__(self,
                 dt=0.001,
                 C_m=200e-12,
                 g_L=10e-9,
                 E_L=-0.070,
                 V_th=-0.050,
                 V_reset=-0.065,
                 t_ref=0.002,
                 E_e=0.0,
                 tau_e=0.005,
                 w_e=1e-9,
                 E_i=-0.080,
                 tau_i=0.010,
                 w_i=1e-9,
                 a=2e-9,
                 b=0.0,
                 tau_w=0.200):
        self.dt = dt
        self.C_m = C_m
        self.g_L = g_L
        self.E_L = E_L
        self.V = E_L
        self.V_th = V_th
        self.V_reset = V_reset
        self.t_ref = t_ref
        self.refrac = 0.0
        # Synaptic
        self.E_e, self.tau_e, self.w_e = E_e, tau_e, w_e
        self.E_i, self.tau_i, self.w_i = E_i, tau_i, w_i
        self.ge = 0.0
        self.gi = 0.0
        # Adaptation
        self.a, self.b, self.tau_w = a, b, tau_w
        self.w = 0.0

    def step(self, spike_exc, spike_inh=0):
        # Update synaptic conductances
        self.ge += self.w_e * spike_exc
        self.gi += self.w_i * spike_inh
        # Exponential decay
        self.ge -= self.dt * (self.ge / self.tau_e)
        self.gi -= self.dt * (self.gi / self.tau_i)

        # Refractory handling
        if self.refrac > 0:
            self.refrac -= self.dt
            self.V = self.V_reset
            return False

        # Currents
        I_L = self.g_L * (self.E_L - self.V)
        I_e = self.ge * (self.E_e - self.V)
        I_i = self.gi * (self.E_i - self.V)
        I_adapt = self.w

        # Voltage update
        dV = (I_L + I_e + I_i - I_adapt) / self.C_m
        self.V += self.dt * dV

        # Adaptation variable (subthreshold + spike-triggered)
        self.w += self.dt * (self.a * (self.V - self.E_L) - self.w) / self.tau_w

        # Spike detection
        if self.V >= self.V_th:
            self.V = self.V_reset
            self.refrac = self.t_ref
            self.w += self.b  # spike-triggered increment
            return True
        return False


# TODO: Bring in elements from ERU Design
# ----------------------------------------------------------------------------
# PrimaryAuditoryCortex network: integrates STRFs, dynamic synapses & LIF neurons
# ----------------------------------------------------------------------------
class PrimaryAuditoryCortex:
    def __init__(self,
                 fs,
                 num_neurons=100,
                 num_freqs=128,
                 num_taps=64,
                 ert_params=None):
        self.fs = fs
        self.dt = 1.0 / fs
        self.units = []
        # ERU design params: use defaults if None
        for _ in range(num_neurons):
            cf = np.random.randint(10, num_freqs - 10)
            bw = np.random.uniform(1.0, 5.0)
            bd = np.random.randint(5, num_taps - 5)
            ts = np.random.uniform(0.01, 0.05)
            mr = np.random.uniform(2.0, 20.0)
            strf = SpectroTemporalReceptiveField(num_freqs, num_taps, cf, bw, bd, ts, mr, fs)
            syn = ShortTermSynapse(U=0.4, tau_rec=0.5, tau_fac=0.0, dt=self.dt)
            neu = ConductanceLIFNeuron(dt=self.dt)
            self.units.append({'strf': strf, 'syn': syn, 'neu': neu})

    def run(self, spectrogram):
        T = spectrogram.shape[1] - self.units[0]['strf'].num_taps + 1
        spikes_out = {i: [] for i in range(len(self.units))}
        # Precompute all drives
        for i, unit in enumerate(self.units):
            drive = unit['strf'].apply(spectrogram)
            # threshold per-unit
            thr = np.percentile(drive, 90)
            input_spikes = drive > thr
            syn_current = unit['syn'].step(input_spikes)
            for t, I_t in enumerate(syn_current[:T]):
                if unit['neu'].step(spike_exc=I_t > 0):
                    spikes_out[i].append(t * self.dt)
        return spikes_out
