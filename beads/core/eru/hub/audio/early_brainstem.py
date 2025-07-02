import numpy as np


class CochlearNucleusCell:
    """
    Base class for cochlear nucleus neurons using an LIF formulation.
    Parameters adapted from Rothman and Manis (2003) models.
    """

    def __init__(self,
                 fs,
                 tau_m=0.002,  # membrane time constant (s)
                 R_m=1e8,  # membrane resistance (Ohm)
                 V_rest=-0.065,  # resting membrane potential (V)
                 V_thresh=-0.050,  # spike threshold (V)
                 V_reset=-0.065,  # reset potential (V)
                 t_ref=0.0005,  # absolute refractory period (s)
                 noise_std=0.0002  # synaptic noise (V)
                 ):
        self.fs = fs
        self.dt = 1.0 / fs
        self.tau_m = tau_m
        self.R_m = R_m
        self.C_m = tau_m / R_m
        self.V_rest = V_rest
        self.V = V_rest
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.t_ref = t_ref
        self.noise_std = noise_std
        self.refrac_time = 0.0

    def step(self, I_syn):
        """
        Advance membrane potential by dt given synaptic input current I_syn (A).
        Returns True on spike.
        """
        if self.refrac_time > 0:
            self.refrac_time -= self.dt
            self.V = self.V_reset
            return False
        # dV/dt = -(V - V_rest)/(R_m*C_m) + I_syn/C_m + noise
        dV = (-(self.V - self.V_rest) / (self.R_m * self.C_m) + I_syn / self.C_m)
        self.V += dV * self.dt + np.random.randn() * self.noise_std * np.sqrt(self.dt)
        if self.V >= self.V_thresh:
            self.V = self.V_reset
            self.refrac_time = self.t_ref
            return True
        return False


class BushyCell(CochlearNucleusCell):
    """
    Spherical bushy cell: preserves timing, high fidelity; low membrane time constant.
    """

    def __init__(self, fs, **kwargs):
        super().__init__(fs, tau_m=0.001, R_m=5e7, t_ref=0.0003, noise_std=0.0001, **kwargs)


class OctopusCell(CochlearNucleusCell):
    """
    Octopus cell: onset detector, high threshold, short tau_m.
    """

    def __init__(self, fs, **kwargs):
        super().__init__(fs, tau_m=0.0005, R_m=1e7, V_thresh=-0.045, t_ref=0.001, noise_std=0.0003, **kwargs)


class ChopperCell(CochlearNucleusCell):
    """
    Chopper cell: regular firing, moderate tau_m, receives sustained input.
    """

    def __init__(self, fs, **kwargs):
        super().__init__(fs, tau_m=0.0025, R_m=8e7, t_ref=0.0004, noise_std=0.0002, **kwargs)


class MSOUnit:
    """
    Medial superior olive model: computes interaural time differences via coincidence detection.
    Uses two input streams of spike trains delayed over range.
    """

    def __init__(self, delays, fs):
        self.delays = delays  # array of ITD delays in seconds
        self.fs = fs
        self.buffers = [np.zeros(int(d * fs)) for d in delays]
        self.threshold = 5  # number of coincident inputs for a spike

    def process(self, spikes_left, spikes_right, duration):
        """
        spikes_left, spikes_right: arrays of spike times (s)
        duration: simulation length (s)
        Returns: array of MSO spike times
        """
        T = int(duration * self.fs)
        count = np.zeros(T, dtype=int)
        # add delayed coincidences
        for buf, d in zip(self.buffers, self.delays):
            # mark left
            for t in spikes_left:
                idx = int((t + d) * self.fs)
                if idx < T:
                    count[idx] += 1
            # mark right with negative delay
            for t in spikes_right:
                idx = int((t - d) * self.fs)
                if 0 <= idx < T:
                    count[idx] += 1
        # detect coincidences
        spikes = np.where(count >= self.threshold)[0] / self.fs
        return spikes


class EarlyBrainstem:
    """
    Pipeline for early brainstem: cochlear nucleus cells + MSO.
    Takes list of AN fiber spike trains per channel and returns processed outputs.
    """

    def __init__(self, fs, num_channels, fibers_per_channel=20):
        self.fs = fs
        self.cn_cells = {
            'bushy': [BushyCell(fs) for _ in range(num_channels)],
            'octopus': [OctopusCell(fs) for _ in range(num_channels)],
            'chopper': [ChopperCell(fs) for _ in range(num_channels)]
        }
        # Example ITDs from -500 to +500 us
        delays = np.linspace(-0.0005, 0.0005, 9)
        self.mso = MSOUnit(delays, fs)

    def run(self, anf_spike_trains, duration):
        """
        anf_spike_trains: list of lists per channel of AN spike arrays.
        duration: simulation time in seconds.
        Returns dict with CN outputs and MSO spikes.
        """
        cn_out = {'bushy': [], 'octopus': [], 'chopper': []}
        # For each channel, feed all fibers into each CN cell type
        for ch_idx in range(len(anf_spike_trains)):
            spikes = np.sort(np.hstack(anf_spike_trains[ch_idx]))
            # convert spike times to current pulses
            cmd = np.zeros(int(duration * self.fs))
            for t in spikes:
                idx = int(t * self.fs)
                if idx < len(cmd):
                    cmd[idx] += 1e-9
            for cell_type, cells in self.cn_cells.items():
                cell = cells[ch_idx]
                out = []
                for i in range(len(cmd)):
                    if cell.step(cmd[i]):
                        out.append(i / self.fs)
                cn_out[cell_type].append(np.array(out))
        # MSO: pair first two channels for binaural
        left = cn_out['bushy'][0]
        right = cn_out['bushy'][1] if len(cn_out['bushy']) > 1 else np.array([])
        mso_spikes = self.mso.process(left, right, duration)
        return {'cochlear_nucleus': cn_out, 'mso': mso_spikes}
