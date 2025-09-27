import numpy as np


class InnerHairCell:
    """
    Biophysically grounded IHC including MET-driven current, membrane RC, and Ca-dependent release rate.
    """

    def __init__(self, segment,
                 g_met=5e-9,  # MET conductance gain (A per m)
                 tau_mem=0.0008,  # membrane time constant (s)
                 v_rest=-0.06,  # resting potential (V)
                 v_thresh=-0.05,  # threshold potential for release (V)
                 ca_exp=3,  # Ca-dependence exponent
                 k_max=5e3):  # max release rate (1/s)
        self.seg = segment
        self.g_met = g_met
        self.tau = tau_mem
        self.v = v_rest + 0.001
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.ca_exp = ca_exp
        self.k_max = k_max
        # spontaneous baseline
        self.v += 0.001  # small depolarization

    def met_current(self):
        # Saturating MET current from BM displacement (m)
        disp = self.seg.displacement
        return self.g_met * np.tanh(disp / 1e-9)

    def function(self, dt):
        # 1) Update membrane potential via RC: dV = (-(V - V_rest) + I_MET*R_input)/tau
        I_met = self.met_current()
        # convert current to equivalent voltage drive (choose R_input to scale)
        dv = (-(self.v - self.v_rest) + I_met * 1e6) * (dt / self.tau)
        self.v += dv
        # 2) Compute release rate k(t)
        vm_eff = max(0.0, self.v - self.v_thresh)
        # power-law Ca dependence
        k = self.k_max * (vm_eff ** self.ca_exp)
        return k


class RibbonSynapse:
    """
    Three-pool Meddis synapse with stochastic release.
    R→C at rate k(t)*R, C→I and loss as per Meddis parameters.
    """

    def __init__(self,
                 R=5,  # initial readily releasable vesicles
                 C=0,
                 I=15,
                 l=30.0,  # loss rate (1/s)
                 r=150.0,  # reuptake rate (1/s)
                 x=100.0,  # mobilization rate (1/s)
                 y=2.0):  # replenishment rate (1/s)
        self.R = R
        self.C = C
        self.I = I

        self.l = l
        self.r = r
        self.x = x
        self.y = y
        self.M = R + I  # max pool size

    def function(self, dt, k):
        # 1) Stochastic release: Binomial draw
        p_rel = min(k * dt, 1.0)
        n_rel = np.random.binomial(int(self.R), p_rel)
        self.R -= n_rel
        self.C += n_rel

        # 2) Deterministic pool flows
        dR = (self.x * self.I + self.y * (self.M - self.R)) * dt
        dC = - (self.l + self.r) * self.C * dt
        dI = (self.r * self.C - self.x * self.I) * dt

        self.R += dR
        self.C += dC
        self.I += dI

        # 3) Clamp non-negative
        self.R, self.C, self.I = map(lambda v: max(v, 0.0), (self.R, self.C, self.I))

        # Return number of vesicles released
        return n_rel


'''
# For each BM segment:
ihc = InnerHairCell(seg)
ribbon = RibbonSynapse()
# In simulation loop:
# k = ihc.update(dt)
# ves = ribbon.update(dt, k)
# ves drives downstream spike generator
'''
