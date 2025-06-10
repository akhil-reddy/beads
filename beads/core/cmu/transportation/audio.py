import numpy as np


class AuditoryNerveFiber:
    """
    Leaky Integrate-and-Fire model for auditory nerve fibers (ANFs), incorporating
    membrane filtering, absolute refractory period, and synaptic input from IHC release.
    """

    def __init__(self,
                 fs,
                 tau_m=0.0005,  # membrane time constant (s) ~0.5 ms
                 R_m=1e7,  # membrane resistance (Ohm) to scale input current
                 C_m=None,  # membrane capacitance (F), if None computed as tau_m/R_m
                 V_rest=-0.065,  # resting potential (V)
                 V_thresh=-0.050,  # spike threshold (V)
                 V_reset=-0.065,  # reset potential after spike (V)
                 t_ref=0.0008,  # absolute refractory period (s) ~0.8 ms
                 noise_std=0.0005  # noise amplitude (V) approximating synaptic variability
                 ):
        self.fs = fs
        self.dt = 1.0 / fs
        self.tau_m = tau_m
        self.R_m = R_m
        self.C_m = C_m if C_m is not None else tau_m / R_m
        self.V_rest = V_rest
        self.V = V_rest
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.t_ref = t_ref
        self.noise_std = noise_std
        self.refrac_time = 0.0  # time remaining in refractory

    def step(self, I_syn):
        """
        Advance the membrane potential by one time step with input current I_syn (A).
        Returns True if a spike occurs at this step.
        """
        if self.refrac_time > 0:
            # during refractory, hold at reset potential
            self.refrac_time -= self.dt
            self.V = self.V_reset
            return False
        # membrane equation: C dV/dt = -(V - V_rest)/R_m + I_syn + noise
        dV = (-(self.V - self.V_rest) / self.R_m + I_syn) * (self.dt / self.C_m)
        # add stochastic component for synaptic noise
        self.V += dV + np.random.randn() * self.noise_std * np.sqrt(self.dt)
        # spike condition
        if self.V >= self.V_thresh:
            # spike!
            self.V = self.V_reset
            self.refrac_time = self.t_ref
            return True
        return False

    def simulate(self, I_syn_array):
        """
        Simulate the ANF for a full I_syn time series (array of synaptic currents in A).
        Returns array of spike times (s).
        """
        spikes = []
        for idx, I in enumerate(I_syn_array):
            if self.step(I):
                spikes.append(idx * self.dt)
        return np.array(spikes)


def generate_spike_trains(vesicle_releases, fs, fiber_params=None):
    """
    Convert each channel’s vesicle release count into spike trains for a population of AN fibers.

    vesicle_releases: list of arrays, where each array gives vesicle counts per time bin
    fs: sampling rate (Hz)
    fiber_params: dict of parameters to pass to AuditoryNerveFiber

    Returns:
      all_spike_trains: list of numpy arrays of spike times for each channel
    """
    all_spike_trains = []
    # default fiber parameters
    fiber_params = fiber_params or {}
    for ves in vesicle_releases:
        # convert vesicle release counts to synaptic current: I = q * N / dt
        # assume vesicle charge q ~ 1e-13 C, so I_syn ~ q * num_releases * fs
        q = fiber_params.get('vesicle_charge', 1e-13)
        I_syn = ves * q * fs
        anf = AuditoryNerveFiber(fs, **fiber_params)
        spikes = anf.simulate(I_syn)
        all_spike_trains.append(spikes)
    return all_spike_trains
