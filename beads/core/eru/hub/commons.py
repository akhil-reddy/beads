import numpy as np
from scipy.integrate import odeint

from beads.core.eru.hub.audio.a1_cortex import ShortTermSynapse


# ----------------------------------------------------------------------------
# Biophysical Multi-Compartment Neuron with Hodgkin-Huxley Channels
# ----------------------------------------------------------------------------
class MultiCompartmentNeuron:
    """
    Two-compartment model: Soma + Dendrite, with Hodgkin-Huxley Na/K channels and adaptation.
    """

    def __init__(self, params):
        # Compartments: 0 = soma, 1 = dendrite
        self.C = params['C']  # [C_soma, C_dend]
        self.g_L = params['g_L']  # leak conductances
        self.E_L = params['E_L']
        # HH channel max conductances (soma only)
        self.g_Na = params['g_Na']
        self.E_Na = params['E_Na']
        self.g_K = params['g_K']
        self.E_K = params['E_K']
        # coupling between compartments
        self.g_c = params['g_c']
        # receptor kinetics containers
        self.receptors = []  # list of receptor objects
        # initial state: V_soma, V_dend, m, h, n
        self.state = np.array([params['E_L'], params['E_L'], 0.05, 0.6, 0.32])

    def add_receptor(self, receptor):
        self.receptors.append(receptor)

    def derivatives(self, y, t):
        V_s, V_d, m, h, n = y
        # Leak currents
        I_L_s = self.g_L[0] * (self.E_L - V_s)
        I_L_d = self.g_L[1] * (self.E_L - V_d)
        # HH currents on soma
        I_Na = self.g_Na * m ** 3 * h * (self.E_Na - V_s)
        I_K = self.g_K * n ** 4 * (self.E_K - V_s)
        # coupling current
        I_c = self.g_c * (V_d - V_s)
        # receptor currents
        I_rec_s = sum([rec.current(V_s, t) for rec in self.receptors])
        I_rec_d = sum([rec.current(V_d, t) for rec in self.receptors])
        # gating variable kinetics (Hodgkin-Huxley)
        alpha_m = 0.1 * (25 - V_s * 1e3) / (np.exp((25 - V_s * 1e3) / 10) - 1)
        beta_m = 4 * np.exp(-V_s * 1e3 / 18)
        alpha_h = 0.07 * np.exp(-V_s * 1e3 / 20)
        beta_h = 1 / (np.exp((30 - V_s * 1e3) / 10) + 1)
        alpha_n = 0.01 * (10 - V_s * 1e3) / (np.exp((10 - V_s * 1e3) / 10) - 1)
        beta_n = 0.125 * np.exp(-V_s * 1e3 / 80)
        dmdt = alpha_m * (1 - m) - beta_m * m
        dhdt = alpha_h * (1 - h) - beta_h * h
        dndt = alpha_n * (1 - n) - beta_n * n
        # voltage derivatives
        dVs_dt = (I_L_s + I_Na + I_K + I_c + I_rec_s) / self.C[0]
        dVd_dt = (I_L_d - I_c + I_rec_d) / self.C[1]
        return [dVs_dt, dVd_dt, dmdt, dhdt, dndt]

    def step(self, tspan):
        # integrate over time window tspan
        sol = odeint(self.derivatives, self.state, tspan)
        self.state = sol[-1]
        return sol[:, 0]  # return soma voltage time series


# ----------------------------------------------------------------------------
# Biophysical Receptor Models: AMPA, NMDA, GABA_A, GABA_B
# ----------------------------------------------------------------------------
class Receptor:
    def __init__(self, g_max, E_rev, tau_rise, tau_decay):
        self.g_max = g_max
        self.E_rev = E_rev
        self.tau_r = tau_rise
        self.tau_d = tau_decay
        self.s = 0.0
        self.x = 0.0

    def event(self):
        # upon presynaptic spike
        self.x += 1.0

    def update(self, dt):
        # double-exponential kinetics
        ds = -self.s / self.tau_d + self.x
        dx = -self.x / self.tau_r
        self.s += ds * dt
        self.x += dx * dt

    def current(self, V, t=None):
        # assume continuous update called externally
        return self.g_max * self.s * (self.E_rev - V)


# ----------------------------------------------------------------------------
# Update ERU classes to use biophysical neurons & receptors
# ----------------------------------------------------------------------------
class BiophysicalERUHub:
    def __init__(self, neuron_params, syn_params, receptor_params):
        self.neuron = MultiCompartmentNeuron(neuron_params)
        # add receptor channels
        for rec_p in receptor_params:
            self.neuron.add_receptor(Receptor(**rec_p))
        self.syn = ShortTermSynapse(**syn_params)
        self.tdt = neuron_params.get('dt', 0.1e-3)

    def step(self, spike_train):
        # synaptic release events
        I_syn = self.syn.step(spike_train)
        # convert synaptic current into receptor events
        # e.g., map I_syn indices to AMPA/NMDA spikes
        tspan = np.arange(0, len(I_syn) * self.tdt, self.tdt)
        V = self.neuron.step(tspan)
        return V
