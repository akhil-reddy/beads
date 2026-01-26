# fast_multicompartment.py
import numpy as np
from numba import njit, prange

# -----------------------
# Helpers (numba-friendly)
# -----------------------
@njit(inline='always')
def safe_vtrap(x):
    # scalar version of vtrap used in HH rates (x is float)
    ax = abs(x)
    if ax < 1e-6:
        return 1.0 - x / 2.0
    else:
        return x / (np.exp(x) - 1.0)


@njit(inline='always')
def mg_block_scalar(voltage_V, voltage_dependent, mg_conc, mg_slope, mg_scale):
    if not voltage_dependent:
        return 1.0
    Vm = voltage_V * 1e3  # V -> mV
    expo = np.exp(-mg_slope * Vm)
    denom = 1.0 + (mg_conc / mg_scale) * expo
    return 1.0 / denom


@njit(inline='always')
def hh_rates(V):  # V in Volts
    Vm = V * 1e3  # mV
    alpha_m = 0.1 * safe_vtrap((25.0 - Vm) / 10.0)
    beta_m = 4.0 * np.exp(-Vm / 18.0)
    alpha_h = 0.07 * np.exp(-Vm / 20.0)
    beta_h = 1.0 / (1.0 + np.exp((30.0 - Vm) / 10.0))
    alpha_n = 0.01 * safe_vtrap((10.0 - Vm) / 10.0)
    beta_n = 0.125 * np.exp(-Vm / 80.0)
    return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n


# -----------------------
# Main batched integrator
# -----------------------
@njit(parallel=True)
def batched_multicompartment_step(
        # neuron state arrays (length N)
        Vs, Vd, m_arr, h_arr, n_arr,
        # receptor state arrays (length R)
        rec_s, rec_x, rec_owner, rec_loc, rec_gmax, rec_Erev, rec_tau_r, rec_tau_d,
        rec_voltage_dependent, rec_mg_conc, rec_mg_slope, rec_mg_scale,
        # neuron-level params (length N or scalars)
        C_s, C_d, g_L_s, g_L_d, E_L, g_Na, E_Na, g_K, E_K, g_c,
        # synaptic input per receptor (length R) -- instantaneous releases (will be added to rec_x before substepping)
        syn_release_per_rec,
        # integration control
        frame_dt, neuron_dt,
        # spike threshold/reset (optional)
        V_thresh, V_reset
):
    """
    Integrate all N neurons for duration `frame_dt` using micro-steps of size `neuron_dt`.
    Arrays:
      - rec_owner: (R,) ints mapping receptor -> neuron idx
      - rec_loc: (R,) int: 0=soma,1=dend,2=both
      - rec_* : receptor params arrays (R,)
    State in/out arrays are modified in place; function returns spikes matrix as (N,) counts during the frame.
    """
    N = Vs.shape[0]
    R = rec_s.shape[0]
    # 1) apply syn_release instantaneously to rec_x
    for ri in prange(R):
        if syn_release_per_rec[ri] != 0.0:
            rec_x[ri] += syn_release_per_rec[ri]

    # number of micro-steps
    n_steps = max(1, int(np.ceil(frame_dt / neuron_dt)))

    # spike counts per neuron (simple integer count of threshold crossings)
    spike_counts = np.zeros(N, dtype=np.int32)

    # micro-stepping loop
    for step in range(n_steps):
        # small dt for this micro-step (last step may be shorter)
        dt = neuron_dt
        if step == n_steps - 1:
            # make sure total integrates precisely to frame_dt
            dt = frame_dt - neuron_dt * (n_steps - 1)
            if dt <= 0.0:
                dt = neuron_dt

        # iterate neurons in parallel
        for i in prange(N):
            # gather local neuron state
            Vs_i = Vs[i]
            Vd_i = Vd[i]
            m = m_arr[i]
            h = h_arr[i]
            n = n_arr[i]

            # 1) compute receptor currents for this neuron by iterating its receptors
            # we must sum I_rec_s and I_rec_d
            I_rec_s = 0.0
            I_rec_d = 0.0

            # For performance, we walk the full receptor list (R) but only process owners == i.
            # Alternatively, you can precompute receptor index ranges per neuron to avoid branching.
            for ri in range(R):
                owner = rec_owner[ri]
                if owner != i:
                    continue
                s = rec_s[ri]
                x = rec_x[ri]
                gmax = rec_gmax[ri]
                E_rev = rec_Erev[ri]
                loc = rec_loc[ri]
                tau_r = rec_tau_r[ri]
                tau_d = rec_tau_d[ri]
                vdep = rec_voltage_dependent[ri]
                mg_c = rec_mg_conc[ri]
                mg_s = rec_mg_slope[ri]
                mg_sc = rec_mg_scale[ri]

                # gating derivatives (we'll advance them below)
                # apply mg block factor
                if loc == 0:  # soma
                    B = mg_block_scalar(Vs_i, vdep, mg_c, mg_s, mg_sc)
                    I_rec_s += gmax * s * B * (E_rev - Vs_i)
                elif loc == 1:  # dend
                    B = mg_block_scalar(Vd_i, vdep, mg_c, mg_s, mg_sc)
                    I_rec_d += gmax * s * B * (E_rev - Vd_i)
                else:  # split half to each
                    B_s = mg_block_scalar(Vs_i, vdep, mg_c, mg_s, mg_sc)
                    B_d = mg_block_scalar(Vd_i, vdep, mg_c, mg_s, mg_sc)
                    I_rec_s += 0.5 * gmax * s * B_s * (E_rev - Vs_i)
                    I_rec_d += 0.5 * gmax * s * B_d * (E_rev - Vd_i)

            # 2) HH currents & leak & coupling
            # Leak currents (I = g*(E - V))
            I_L_s = g_L_s * (E_L - Vs_i)
            I_L_d = g_L_d * (E_L - Vd_i)

            # HH rates on soma
            alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = hh_rates(Vs_i)
            dmdt = alpha_m * (1.0 - m) - beta_m * m
            dhdt = alpha_h * (1.0 - h) - beta_h * h
            dndt = alpha_n * (1.0 - n) - beta_n * n

            I_Na = g_Na * (m ** 3) * h * (E_Na - Vs_i)
            I_K = g_K * (n ** 4) * (E_K - Vs_i)

            I_c = g_c * (Vd_i - Vs_i)

            # 3) membrane derivatives
            dVs_dt = (I_L_s + I_Na + I_K + I_c + I_rec_s) / C_s
            dVd_dt = (I_L_d - I_c + I_rec_d) / C_d

            # 4) advance gating variables with Euler
            m += dmdt * dt
            h += dhdt * dt
            n += dndt * dt

            # 5) advance voltages (Euler)
            Vs_i += dVs_dt * dt
            Vd_i += dVd_dt * dt

            # 6) Advance receptor gating variables for the receptors of neuron i
            for ri in range(R):
                owner = rec_owner[ri]
                if owner != i:
                    continue
                s = rec_s[ri]
                x = rec_x[ri]
                tau_r = rec_tau_r[ri]
                tau_d = rec_tau_d[ri]
                # ds/dt = -s/tau_d + x
                dsdt = -s / tau_d + x
                dxdt = -x / tau_r
                s += dsdt * dt
                x += dxdt * dt
                # clamp non-negative where appropriate
                if s < 0.0: s = 0.0
                if x < 0.0: x = 0.0
                rec_s[ri] = s
                rec_x[ri] = x

            # 7) write back neuron state
            Vs[i] = Vs_i
            Vd[i] = Vd_i
            m_arr[i] = m
            h_arr[i] = h
            n_arr[i] = n

            # 8) thresholding/spike counting (very simple)
            if Vs_i >= V_thresh:
                spike_counts[i] += 1
                Vs[i] = V_reset  # reset after spike

    return spike_counts
