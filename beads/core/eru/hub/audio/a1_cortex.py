import numpy as np

from beads.core.eru.hub.commons import SpectroTemporalReceptiveField
from beads.core.eru.interneuron import ShortTermSynapse, MultiCompartmentNeuron, Receptor


# ----------------------------------------------------------------------------
# PrimaryAuditoryCortex using MultiCompartmentNeuron + receptors
# ----------------------------------------------------------------------------
class PrimaryAuditoryCortex:
    def __init__(self,
                 fs,
                 num_neurons=100,
                 num_freqs=128,
                 num_taps=64,
                 neuron_params=None,
                 receptor_params=None,
                 syn_params=None):
        self.fs = fs
        self.dt = 1.0 / fs
        self.units = []
        # defaults
        if neuron_params is None:
            neuron_params = {
                'C': [200e-12, 200e-12],
                'g_L': [10e-9, 10e-9],
                'E_L': -65e-3,
                'g_Na': 1200e-9,
                'E_Na': 50e-3,
                'g_K': 360e-9,
                'E_K': -77e-3,
                'g_c': 5e-9,
                'dt': self.dt,
                'V0': -65e-3
            }
        if receptor_params is None:
            receptor_params = [
                {'g_max': 1e-9, 'E_rev': 0.0, 'tau_rise': 0.001, 'tau_decay': 0.005, 'location': 'soma',
                 'name': 'AMPA', 'voltage_dependent': False},
                {'g_max': 0.5e-9, 'E_rev': 0.0, 'tau_rise': 0.005, 'tau_decay': 0.080, 'location': 'dend',
                 'name': 'NMDA', 'voltage_dependent': True}
            ]
        if syn_params is None:
            syn_params = {'U': 0.4, 'tau_rec': 0.5, 'tau_fac': 0.0, 'dt': self.dt}

        # instantiate units
        for _ in range(num_neurons):
            cf = np.random.randint(10, num_freqs - 10)
            bw = np.random.uniform(1.0, 5.0)
            bd = np.random.randint(5, num_taps - 5)
            ts = np.random.uniform(0.01, 0.05)
            mr = np.random.uniform(2.0, 20.0)
            strf = SpectroTemporalReceptiveField(num_freqs, num_taps, cf, bw, bd, ts, mr, fs)
            syn = ShortTermSynapse(**syn_params)
            neu = MultiCompartmentNeuron(neuron_params.copy())
            # add receptors to neuron (deepcopy-like)
            for rp in receptor_params:
                rec = Receptor(**rp)
                neu.add_receptor(rec)
            self.units.append({'strf': strf, 'syn': syn, 'neu': neu})

    def run(self, spectrogram):
        """
        spectrogram: shape (num_freqs, T)
        returns dict {unit_idx: spike_time_array (s)}
        """
        spikes_out = {}
        for i, unit in enumerate(self.units):
            drive = unit['strf'].apply(spectrogram)  # length Tdrive
            # convert drive -> presynaptic spike counts (Poisson link)
            # scale drive to reasonable firing probability; use soft rectification
            drive_pos = np.maximum(drive, 0.0)
            # convert amplitude to instantaneous rate parameter (this scaling is arbitrary and tunable)
            lam = drive_pos / (np.max(drive_pos) + 1e-12) * 20.0  # firing rate up to ~20 spikes/s
            p = 1.0 - np.exp(-lam * self.dt)  # per-bin probability
            prespikes = (np.random.rand(p.size) < p).astype(float)

            # short-term synapse releases (1D)
            rel = unit['syn'].step(prespikes)  # (Tdrive,)
            # map release to receptor channels (AMPA,NMDA). Adjust weights as needed.
            # Here AMPA gets 0.8, NMDA gets 0.5 * rel (slower but smaller amplitude)
            syn_release = np.stack([rel * 0.8, rel * 0.5], axis=1)  # shape (Tdrive, 2)

            # integrate neuron; returns times (absolute seconds) and soma voltage trace
            times, Vs = unit['neu'].step(syn_release, t0=0.0)

            # detect spikes from Vs via threshold crossing (simple)
            V_thresh = -30e-3  # −30 mV threshold for detecting an action potential peak
            spike_idx = np.where(Vs >= V_thresh)[0]
            # collapse indices to unique events (rising edges)
            if spike_idx.size > 0:
                spikes_bins = np.unique(spike_idx)
                spike_times = (times[spikes_bins]).tolist()
            else:
                spike_times = []
            spikes_out[i] = np.array(spike_times)
        return spikes_out
