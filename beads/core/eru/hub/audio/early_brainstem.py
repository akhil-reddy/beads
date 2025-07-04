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

    def __init__(self, delays, fs, threshold=5):
        self.delays = delays  # array of ITD delays in seconds
        self.fs = fs
        self.threshold = threshold

    def process(self, spikes_left, spikes_right, duration):
        """
        spikes_left, spikes_right: arrays of spike times (s)
        duration: simulation length (s)
        Returns: array of MSO spike times
        """
        T = int(duration * self.fs)
        count = np.zeros(T, dtype=int)
        # add delayed coincidences
        for d in self.delays:
            for t in spikes_left:
                idx = int((t + d) * self.fs)
                if idx < T:
                    count[idx] += 1
            for t in spikes_right:
                idx = int((t - d) * self.fs)
                if 0 <= idx < T:
                    count[idx] += 1
        spikes = np.where(count >= self.threshold)[0] / self.fs
        return spikes


class LSOUnit:
    """
    Lateral superior olive model: computes interaural level differences via excitation-inhibition comparison.
    """

    def __init__(self, fs, threshold=3):
        self.fs = fs
        self.threshold = threshold

    def process(self, exc_spikes, inh_spikes, duration):
        """
        exc_spikes: array of excitatory spike times (s)
        inh_spikes: array of inhibitory spike times (s)
        duration: simulation length (s)
        Returns: array of LSO spike times
        """
        T = int(duration * self.fs)
        exc_count = np.zeros(T, dtype=int)
        inh_count = np.zeros(T, dtype=int)
        for t in exc_spikes:
            idx = int(t * self.fs)
            if idx < T:
                exc_count[idx] += 1
        for t in inh_spikes:
            idx = int(t * self.fs)
            if idx < T:
                inh_count[idx] += 1
        diff = exc_count - inh_count
        spikes = np.where(diff >= self.threshold)[0] / self.fs
        return spikes


class EarlyBrainstem:
    """
    Pipeline for early brainstem: cochlear nucleus cells + MSO & LSO + efferent feedback stub.
    Input: list of AN fiber spike trains per channel. Output: dict of CN, MSO, LSO, and feedback gain.
    """

    def __init__(self, fs, num_channels, fibers_per_channel=20):
        self.fs = fs
        self.cn_cells = {
            'bushy': [BushyCell(fs) for _ in range(num_channels)],
            'octopus': [OctopusCell(fs) for _ in range(num_channels)],
            'chopper': [ChopperCell(fs) for _ in range(num_channels)]
        }
        # ITDs from -500 to +500 us
        delays = np.linspace(-0.0005, 0.0005, 9)
        self.mso = MSOUnit(delays, fs)
        self.lso = LSOUnit(fs)
        self.efferent_gain = 1.0  # placeholder for OHC feedback

    def efferent_feedback(self, mso_spikes, lso_spikes):
        """
        Stub for efferent feedback: adjusts OHC gain based on binaural contrast.
        """
        # Example: increase gain if spatial cues weak, decrease if strong
        spatial_strength = len(mso_spikes) + len(lso_spikes)
        self.efferent_gain = 1.0 + 0.1 * np.tanh(1.0 / (1 + spatial_strength))
        return self.efferent_gain

    def run(self, anf_spike_trains, duration):
        """
        anf_spike_trains: list of lists per channel of AN spike arrays.
        duration: simulation time in seconds.
        Returns dict with cochlear nucleus outputs, MSO, LSO, and updated efferent gain.
        """
        cn_out = {ctype: [] for ctype in self.cn_cells}
        # Process each channel through CN
        for ch_idx, fibers in enumerate(anf_spike_trains):
            spikes = np.sort(np.hstack(fibers))
            cmd = np.zeros(int(duration * self.fs))
            for t in spikes:
                idx = int(t * self.fs)
                if idx < len(cmd):
                    cmd[idx] += 1e-9 * self.efferent_gain
            for ctype, cells in self.cn_cells.items():
                cell = cells[ch_idx]
                out_times = []
                for i, I in enumerate(cmd):
                    if cell.step(I):
                        out_times.append(i / self.fs)
                cn_out[ctype].append(np.array(out_times))
        # Binaural processing on bushy outputs of first two channels
        left = cn_out['bushy'][0]
        right = cn_out['bushy'][1] if len(cn_out['bushy']) > 1 else np.array([])
        mso_spikes = self.mso.process(left, right, duration)
        lso_spikes = self.lso.process(left, right, duration)
        # Update efferent gain
        new_gain = self.efferent_feedback(mso_spikes, lso_spikes)
        return {
            'cochlear_nucleus': cn_out,
            'mso': mso_spikes,
            'lso': lso_spikes,
            'efferent_gain': new_gain
        }
