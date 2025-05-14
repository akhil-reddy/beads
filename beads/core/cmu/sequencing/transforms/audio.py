import numpy as np
from scipy.signal import butter, filtfilt


class InnerHairCell:
    """
    Inner hair cell transduction:
      1. Mechano-electrical transduction (half-wave + low-pass) :contentReference[oaicite:0]{index=0}
      2. RibbonSynapse integration
    """

    def __init__(self, bm_segment, ihc_cutoff=1000.0):
        self.seg = bm_segment
        # Design 1st-order lowpass filter (~1 kHz) for receptor potential
        b, a = butter(1, ihc_cutoff / (bm_segment.seg_width if False else bm_segment.seg_width), fs=1.0)
        self.b, self.a = b, a
        self.v_m = 0.0  # receptor potential (V)

    def mechanoelectrical(self, dt):
        """
        Simple MET: half-wave rectify BM velocity → receptor current → low-pass
        Equivalent to IHC MET kinetics :contentReference[oaicite:1]{index=1}.
        """
        vel = self.seg.velocity
        current = max(0.0, vel)  # only excitatory deflections
        # integrate RC (here using filtfilt for simplicity)
        self.v_m = filtfilt(self.b, self.a, np.array([self.v_m, current]))[-1]
        return self.v_m

    def step(self, dt, ribbon_synapse):
        """
        1) Compute receptor potential
        2) Drive ribbon synapse release (via v_m → Ca²⁺ proxy)
        """
        vm = self.mechanoelectrical(dt)
        # Use vm as proportional to Ca²⁺ influx (linear model) :contentReference[oaicite:2]{index=2}
        ca_signal = max(0.0, vm)
        release = ribbon_synapse.step(dt, ca_signal)
        return release


class RibbonSynapse:
    """
    Three-pool Meddis synapse model (R, C, I):
      R: readily releasable
      C: synaptic cleft
      I: recycling pool
    Meddis et al. (1990) parameters :contentReference[oaicite:5]{index=5}.
    """

    def __init__(self,
                 R=20.0, t_R=1e-3,
                 C=0.0, t_C=0.5e-3,
                 I=80.0, t_I=20.0e-3):
        # Vesicle pools (in arbitrary units)
        self.R = R  # ready
        self.C = C  # cleft
        self.I = I  # inactive/recycling
        # Time constants (s)
        self.t_R = t_R  # replenishment into R
        self.t_C = t_C  # clearance from cleft
        self.t_I = t_I  # recycling to R
        # Release rate constant per Ca signal
        self.k_rel = 1.0e3  # (1/s per unit Ca) :contentReference[oaicite:6]{index=6}

    def step(self, dt, ca):
        """
        Update vesicle pools and return neurotransmitter in cleft ⟶ AN spike drive.
        """
        # Release from R into C
        r_rel = self.k_rel * ca * self.R
        # Pool updates (Euler)
        dR = (self.t_I ** -1) * self.I + (self.t_R ** -1) - r_rel
        dC = r_rel - (self.C / self.t_C)
        dI = (self.C / self.t_C) - (self.I / self.t_I)
        self.R += dR * dt
        self.C += dC * dt
        self.I += dI * dt
        # Clamp pools to ≥0
        self.R, self.C, self.I = map(lambda x: max(0.0, x), (self.R, self.C, self.I))
        # Return cleft transmitter as release signal
        return self.C


'''
# assuming bm = BasilarMembrane(); ohcs created; now create IHCs & synapses

ihcs = []
ribbons = []
for seg in bm.segments:
    syn = RibbonSynapse()
    ribbon = syn
    ihc = InnerHairCell(seg)
    ihcs.append(ihc)
    ribbons.append(ribbon)

# In simulation loop:
dt = 1e-6
for t in np.arange(0, 0.1, dt):
    # 1) BM passive + OHC active (from previous steps)
    bm.apply_pressure(pressure_spectrum)  
    ohc.step(dt, synaptic_input)  
    bm.step(dt)
    # 2) IHC & ribbon
    for ihc, ribbon in zip(ihcs, ribbons):
        nt = ihc.step(dt, ribbon)
        # nt drives spiral ganglion → spikes...
'''
