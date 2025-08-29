import numpy as np
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------------
# Utilities: safe vtrap for HH rates
# ----------------------------------------------------------------------------
def vtrap(x):
    """Stable x/(exp(x)-1) evaluator for small x (works elementwise)."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-6
    out = np.empty_like(x, dtype=float)
    out[small] = 1.0 - x[small] / 2.0
    out[~small] = x[~small] / (np.exp(x[~small]) - 1.0)
    return out


# ----------------------------------------------------------------------------
# Receptor (double exponential) with optional NMDA Mg2+ voltage dependence
# ----------------------------------------------------------------------------
class Receptor:
    def __init__(self, g_max, E_rev, tau_rise, tau_decay,
                 location='soma', name=None,
                 voltage_dependent=False, mg_conc=1.0, mg_slope=0.062, mg_scale=3.57):
        """
        location: 'soma' or 'dend' (where the conductance acts)
        voltage_dependent: if True, apply Mg2+ block using Jahr & Stevens form
        mg_conc: extracellular Mg2+ in mM (default 1.0)
        mg_slope: slope constant (default 0.062 per mV)
        mg_scale: scaling constant in denominator (default 3.57 mM)
        """
        self.g_max = float(g_max)
        self.E_rev = float(E_rev)
        self.tau_r = float(tau_rise)
        self.tau_d = float(tau_decay)
        self.location = location
        self.name = name or f"Receptor_{id(self)}"

        # NMDA Mg block params
        self.voltage_dependent = bool(voltage_dependent)
        self.mg_conc = float(mg_conc)
        self.mg_slope = float(mg_slope)
        self.mg_scale = float(mg_scale)

        # initial gating variables (placeholders - packed into neuron state)
        self.s0 = 0.0
        self.x0 = 0.0

    def mg_block(self, V):
        """
        Compute Mg2+ blocking factor B(V) in [0,1].
        V: membrane voltage in volts (internal unit). Convert to mV for formula.
        Uses the form: B(V) = 1 / (1 + (Mg/scale) * exp(-slope * Vm))
        where Vm is in mV.
        """
        if not self.voltage_dependent:
            return 1.0
        Vm = V * 1e3  # convert V -> mV
        # scalar safe evaluation
        # cap exponent to avoid overflow for extremely negative Vm
        expo = np.exp(-self.mg_slope * Vm)
        denom = 1.0 + (self.mg_conc / self.mg_scale) * expo
        return 1.0 / denom

    def __repr__(self):
        vd = "NMDA-like" if self.voltage_dependent else "non-voltage-dependent"
        return f"<Receptor {self.name} {vd} loc={self.location} gmax={self.g_max}>"


# TODO: Bring in elements from ERU Design
# ----------------------------------------------------------------------------
# Two-Compartment Hodgkin-Huxley Neuron with receptors embedded in ODE state
# ----------------------------------------------------------------------------
class MultiCompartmentNeuron:
    """
    Two-compartment neuron (soma + dendrite). Hodgkin-Huxley channels on soma.
    Receptor gating variables (s, x) are included in the ODE state vector so they
    are integrated consistently with membrane voltages and HH gates.
    Units: volt (V) for voltages internally; time in seconds.
    """

    def __init__(self, params):
        self.C = np.array(params['C'], dtype=float)  # [C_soma, C_dend] in Farads
        self.g_L = np.array(params['g_L'], dtype=float)  # [gL_soma, gL_dend] in Siemens
        self.E_L = float(params['E_L'])  # leak reversal (V)
        self.g_Na = float(params['g_Na'])  # soma Na max conductance (S)
        self.E_Na = float(params['E_Na'])  # Na reversal (V)
        self.g_K = float(params['g_K'])  # soma K max conductance (S)
        self.E_K = float(params['E_K'])  # K reversal (V)
        self.g_c = float(params['g_c'])  # coupling conductance (S)
        self.dt = params.get('dt', 0.1e-4)  # integration bin in seconds

        V0 = params.get('V0', self.E_L)
        m0 = params.get('m0', 0.05)
        h0 = params.get('h0', 0.6)
        n0 = params.get('n0', 0.32)

        self.receptors = []
        self.base_state0 = np.array([V0, V0, m0, h0, n0], dtype=float)

    def add_receptor(self, receptor: Receptor):
        self.receptors.append(receptor)

    # HH rate functions
    def _hh_rates(self, V):
        Vm = V * 1e3  # mV
        alpha_m = 0.1 * vtrap((25.0 - Vm) / 10.0)
        beta_m = 4.0 * np.exp(-Vm / 18.0)
        alpha_h = 0.07 * np.exp(-Vm / 20.0)
        beta_h = 1.0 / (1.0 + np.exp((30.0 - Vm) / 10.0))
        alpha_n = 0.01 * vtrap((10.0 - Vm) / 10.0)
        beta_n = 0.125 * np.exp(-Vm / 80.0)
        return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

    def derivatives(self, t, y):
        Nrec = len(self.receptors)
        Vs = y[0]
        Vd = y[1]
        m = y[2]
        h = y[3]
        n = y[4]

        rec_vals = y[5:] if Nrec > 0 else np.array([])
        if Nrec > 0:
            s = rec_vals[0::2]
            x = rec_vals[1::2]
        else:
            s = np.array([])
            x = np.array([])

        # Leak
        I_L_s = self.g_L[0] * (self.E_L - Vs)
        I_L_d = self.g_L[1] * (self.E_L - Vd)

        # HH soma currents
        I_Na = self.g_Na * (m ** 3) * h * (self.E_Na - Vs)
        I_K = self.g_K * (n ** 4) * (self.E_K - Vs)

        # coupling
        I_c = self.g_c * (Vd - Vs)

        # receptor currents with possible voltage dependence (NMDA Mg block)
        I_rec_s = 0.0
        I_rec_d = 0.0
        for idx, rec in enumerate(self.receptors):
            si = s[idx] if Nrec > 0 else 0.0
            if rec.location == 'soma':
                factor = rec.mg_block(Vs)
                I_rec_s += rec.g_max * si * factor * (rec.E_rev - Vs)
            elif rec.location == 'dend':
                factor = rec.mg_block(Vd)
                I_rec_d += rec.g_max * si * factor * (rec.E_rev - Vd)
            else:
                # split
                factor_s = rec.mg_block(Vs)
                factor_d = rec.mg_block(Vd)
                I_rec_s += 0.5 * rec.g_max * si * factor_s * (rec.E_rev - Vs)
                I_rec_d += 0.5 * rec.g_max * si * factor_d * (rec.E_rev - Vd)

        # gating derivatives
        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = self._hh_rates(Vs)
        dmdt = alpha_m * (1.0 - m) - beta_m * m
        dhdt = alpha_h * (1.0 - h) - beta_h * h
        dndt = alpha_n * (1.0 - n) - beta_n * n

        dVs_dt = (I_L_s + I_Na + I_K + I_c + I_rec_s) / self.C[0]
        dVd_dt = (I_L_d - I_c + I_rec_d) / self.C[1]

        rec_derivs = []
        for idx, rec in enumerate(self.receptors):
            si = s[idx]
            xi = x[idx]
            dsdt = -si / rec.tau_d + xi
            dxdt = -xi / rec.tau_r
            rec_derivs.extend([dsdt, dxdt])

        dy = [dVs_dt, dVd_dt, dmdt, dhdt, dndt] + rec_derivs
        return np.array(dy, dtype=float)

    def pack_state(self, base_state=None, rec_states=None):
        if base_state is None:
            base_state = self.base_state0
        if rec_states is None:
            rec_states = []
            for rec in self.receptors:
                rec_states.extend([rec.s0, rec.x0])
        return np.concatenate([base_state, np.array(rec_states, dtype=float)])

    def unpack_state(self, y):
        Nrec = len(self.receptors)
        base = y[:5]
        rec_vals = []
        if Nrec > 0:
            rv = y[5:]
            for i in range(Nrec):
                rec_vals.append((rv[2 * i], rv[2 * i + 1]))
        return base, rec_vals

    def step(self, syn_release, t0=0.0):
        syn_release = np.asarray(syn_release)
        T = syn_release.shape[0]
        Nrec = len(self.receptors)
        if Nrec == 0:
            raise RuntimeError("No receptors added to neuron. Add with add_receptor().")

        if syn_release.ndim == 1:
            syn_release = np.tile(syn_release[:, None], (1, Nrec))
        elif syn_release.shape[1] != Nrec:
            try:
                syn_release = np.broadcast_to(syn_release, (T, Nrec))
            except Exception:
                raise ValueError("syn_release second dimension must match number of receptors")

        dt = self.dt
        tspan_bins = t0 + np.arange(0.0, T * dt, dt)
        times = []
        Vs_trace = []
        y = self.pack_state()

        for i in range(T):
            t_start = tspan_bins[i]
            t_end = t_start + dt

            # apply release events -> increment x for each receptor
            for j, rec in enumerate(self.receptors):
                idx_x = 5 + 2 * j + 1
                y[idx_x] += float(syn_release[i, j])

            sol = solve_ivp(fun=self.derivatives, t_span=(t_start, t_end), y0=y,
                            method='BDF', atol=1e-8, rtol=1e-6, max_step=dt / 4.0)

            y = sol.y[:, -1]
            times.append(t_end)
            Vs_trace.append(y[0])

        return np.array(times), np.array(Vs_trace)


# TODO: Bring in elements from ERU Design
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


if __name__ == "__main__":
    # tiny example: 100 ms simulation, 0.1 ms bins
    dt = 0.1e-3
    T = int(0.1 / dt)  # 100 ms
    spike_train = np.zeros(T)
    spike_train[5] = 1  # single presynaptic spike at bin 5

    neuron_params = {
        'C': [200e-12, 200e-12],
        'g_L': [10e-9, 10e-9],
        'E_L': -65e-3,
        'g_Na': 1200e-9,
        'E_Na': 50e-3,
        'g_K': 360e-9,
        'E_K': -77e-3,
        'g_c': 5e-9,
        'dt': dt,
        'V0': -65e-3
    }

    receptor_params = [
        # AMPA: fast, soma, non-voltage-dependent
        {'g_max': 1e-9, 'E_rev': 0.0, 'tau_rise': 0.001, 'tau_decay': 0.005,
         'location': 'soma', 'name': 'AMPA', 'voltage_dependent': False},
        # NMDA: slow, dend, voltage-dependent Mg block enabled
        {'g_max': 0.5e-9, 'E_rev': 0.0, 'tau_rise': 0.005, 'tau_decay': 0.080,
         'location': 'dend', 'name': 'NMDA', 'voltage_dependent': True, 'mg_conc': 1.0,
         'mg_slope': 0.062, 'mg_scale': 3.57}
    ]

    syn_params = {'n_rec': len(receptor_params)}
