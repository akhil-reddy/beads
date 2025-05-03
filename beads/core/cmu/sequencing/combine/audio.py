import numpy as np


class BasilarMembrane:
    class Segment:
        def __init__(self, cf, width_mm, stiffness, mass):
            self.cf = cf
            self.width_mm = width_mm
            self.stiffness = stiffness  # N/m
            self.mass = mass  # kg
            self.velocity = 0.0  # m/s
            self.displacement = 0.0  # m
            self.force = 0.0  # N

    def __init__(self, n=200, cf_range=(20, 20000)):
        cfs = np.logspace(np.log10(cf_range[1]), np.log10(cf_range[0]), n)
        widths = np.linspace(8e-5, 6.5e-4, n)  # m
        stiffness = 1.0 / widths  # N/m approx inverse
        mass = np.full(n, 1e-9)  # kg, small
        self.segments = [BasilarMembrane.Segment(cf, w, k, m)
                         for cf, w, k, m in zip(cfs, widths, stiffness, mass)]

    def apply_pressure(self, pressure_spectrum):
        for seg in self.segments:
            p = pressure_spectrum.get(seg.cf, 0.0)
            seg.force += p * seg.width_mm  # force ~ pressure×area

    def step(self, dt):
        for seg in self.segments:
            # integrate simple mass-spring
            acc = (seg.force - seg.stiffness * seg.displacement) / seg.mass
            seg.velocity += acc * dt
            seg.displacement += seg.velocity * dt
            seg.force = 0.0


class OuterHairCell:
    """
    Prestin-based OHC electromotility with Boltzmann kinetics,
    anion modulation, and efferent inhibition.
    """

    def __init__(self, segment,
                 z_max=1.0e-8,  # max length change (m)
                 V_half=-0.05,  # voltage at half-max (V)
                 z_slope=0.02,  # slope factor (V)
                 cl_sensitivity=0.1,  # shift per [Cl-] (mV)
                 inh_gain=1.0):
        self.seg = segment
        self.z_max = z_max
        self.V_half = V_half
        self.z_slope = z_slope
        self.cl_sens = cl_sensitivity
        self.inh_gain = inh_gain
        self.Vm = -0.04  # resting potential (V)
        self.Cl = 140.0  # intracellular Cl- (mM)
        self.eff = 0.0  # efferent inhibition [0,1]

    def prestin_fraction(self):
        # Two-state Boltzmann for prestin charge movement
        Veff = self.Vm - (self.cl_sens * (self.Cl - 140) / 1000.0)  # Cl shifts Vhalf
        return 1.0 / (1 + np.exp(-(Veff - self.V_half) / self.z_slope))

    def electromechanical_force(self):
        # Force ~ stiffness × OHC length change
        frac = self.prestin_fraction() * (1 - self.eff * self.inh_gain)
        dz = (frac * 2 - 1) * self.z_max  # length change ±z_max
        return self.seg.stiffness * dz

    def step(self, dt, synaptic_input):
        # synaptic_input: current injection (A) from mechano‑transducer
        Cm = 10e-12  # membrane capacitance (F)
        # simple RC update Vm
        dv = (- (self.Vm + 0.06) / 50e6 + synaptic_input / Cm) * dt
        self.Vm += dv
        # apply force to BM
        self.seg.force += self.electromechanical_force()


class MedialOlivocochlear:
    """
    Efferent fibers release ACh/GABA onto OHCs to inhibit prestin.
    """

    def __init__(self, ohcs):
        self.ohcs = ohcs

    def stimulate(self, rate):
        # rate: 0–100% release → sets efferent level on each OHC
        for ohc in self.ohcs:
            ohc.eff = min(1.0, rate)
