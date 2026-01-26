# helper_flatten.py
import numpy as np


def flatten_neurons_and_receptors(neuron_list):
    N = len(neuron_list)
    # collect neuron params
    Vs = np.empty(N, dtype=np.float64)
    Vd = np.empty(N, dtype=np.float64)
    m = np.empty(N, dtype=np.float64)
    h = np.empty(N, dtype=np.float64)
    n = np.empty(N, dtype=np.float64)

    # collect neurons' per-param arrays (use same naming as kernel)
    C_s = np.empty(N, dtype=np.float64)
    C_d = np.empty(N, dtype=np.float64)
    g_L_s = np.empty(N, dtype=np.float64)
    g_L_d = np.empty(N, dtype=np.float64)
    # single-valued across neurons? pick scalar if identical
    E_L = neuron_list[0].E_L
    g_Na = neuron_list[0].g_Na
    E_Na = neuron_list[0].E_Na
    g_K = neuron_list[0].g_K
    E_K = neuron_list[0].E_K
    g_c = neuron_list[0].g_c

    rec_owner = []
    rec_s = []
    rec_x = []
    rec_gmax = []
    rec_Erev = []
    rec_tau_r = []
    rec_tau_d = []
    rec_loc = []
    rec_vdep = []
    rec_mg_c = []
    rec_mg_s = []
    rec_mg_sc = []

    for i, neu in enumerate(neuron_list):
        # unpack base state
        base, rec_states = neu.unpack_state(neu.pack_state())
        Vs[i] = base[0]
        Vd[i] = base[1]
        m[i] = base[2]
        h[i] = base[3]
        n[i] = base[4]

        C_s[i] = neu.C[0]
        C_d[i] = neu.C[1]
        g_L_s[i] = neu.g_L[0]
        g_L_d[i] = neu.g_L[1]

        # receptors
        for ridx, rec in enumerate(neu.receptors):
            rec_owner.append(i)
            rec_s.append(rec.s0)
            rec_x.append(rec.x0)
            rec_gmax.append(rec.g_max)
            rec_Erev.append(rec.E_rev)
            rec_tau_r.append(rec.tau_r)
            rec_tau_d.append(rec.tau_d)
            # location encoding
            if rec.location == 'soma':
                rec_loc.append(0)
            elif rec.location == 'dend':
                rec_loc.append(1)
            else:
                rec_loc.append(2)
            rec_vdep.append(1 if rec.voltage_dependent else 0)
            rec_mg_c.append(rec.mg_conc)
            rec_mg_s.append(rec.mg_slope)
            rec_mg_sc.append(rec.mg_scale)

    rec_owner = np.array(rec_owner, dtype=np.int32)
    rec_s = np.array(rec_s, dtype=np.float64)
    rec_x = np.array(rec_x, dtype=np.float64)
    rec_gmax = np.array(rec_gmax, dtype=np.float64)
    rec_Erev = np.array(rec_Erev, dtype=np.float64)
    rec_tau_r = np.array(rec_tau_r, dtype=np.float64)
    rec_tau_d = np.array(rec_tau_d, dtype=np.float64)
    rec_loc = np.array(rec_loc, dtype=np.int32)
    rec_vdep = np.array(rec_vdep, dtype=np.int8)
    rec_mg_c = np.array(rec_mg_c, dtype=np.float64)
    rec_mg_s = np.array(rec_mg_s, dtype=np.float64)
    rec_mg_sc = np.array(rec_mg_sc, dtype=np.float64)

    return {
        'Vs': Vs, 'Vd': Vd, 'm': m, 'h': h, 'n': n,
        'C_s': C_s, 'C_d': C_d, 'g_L_s': g_L_s, 'g_L_d': g_L_d,
        'E_L': E_L, 'g_Na': g_Na, 'E_Na': E_Na, 'g_K': g_K, 'E_K': E_K, 'g_c': g_c,
        'rec_owner': rec_owner, 'rec_s': rec_s, 'rec_x': rec_x,
        'rec_gmax': rec_gmax, 'rec_Erev': rec_Erev,
        'rec_tau_r': rec_tau_r, 'rec_tau_d': rec_tau_d,
        'rec_loc': rec_loc, 'rec_voltage_dependent': rec_vdep,
        'rec_mg_conc': rec_mg_c, 'rec_mg_slope': rec_mg_s, 'rec_mg_scale': rec_mg_sc
    }
