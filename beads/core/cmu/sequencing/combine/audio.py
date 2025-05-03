import numpy as np


class BasilarMembrane:
    """
    Represents the passive tonotopic mechanics of the basilar membrane.
    Each segment has a characteristic frequency (CF), width, and stiffness.
    """

    class Segment:
        def __init__(self, cf, width_mm, stiffness_Nm2):
            self.cf = cf  # characteristic freq (Hz)
            self.width_mm = width_mm  # width at this place (mm)
            self.stiffness = stiffness_Nm2  # stiffness (N/m^2)
            self.displacement = 0.0  # current displacement (μm)

    def __init__(self, n_segments=100, cf_range=(20, 20000)):
        """
        Create a tonotopic array of BM segments from base→apex.
        Width increases and stiffness decreases apically.
        """
        self.segments = []
        # log-spaced CFs from high (base) to low (apex)
        cfs = np.logspace(np.log10(cf_range[1]), np.log10(cf_range[0]), n_segments)
        # width from 0.08 mm (base) to 0.65 mm (apex)
        widths = np.linspace(0.08, 0.65, n_segments)
        # stiffness inversely proportional to width (approx)
        stiffness = 1.0 / widths
        for cf, w, k in zip(cfs, widths, stiffness):
            self.segments.append(BasilarMembrane.Segment(cf, w, k))

    def stimulate(self, input_pressure_spectrum):
        """
        Passively displace each segment according to input spectrum.
        input_pressure_spectrum: dict {cf: pressure_amplitude}
        """
        for seg in self.segments:
            # simple resonant response: disp ∝ pressure / stiffness at matching CF
            p = input_pressure_spectrum.get(seg.cf, 0.0)
            seg.displacement = p / seg.stiffness


class OuterHairCell:
    """
    Models an OHC that (a) electromotile amplifies BM displacement, (b) can be inhibited
    by efferent neurotransmitter (ACh/GABA), and (c) feeds back onto the BM segment.
    """

    def __init__(self, bm_segment, prestin_gain=50.0, inh_sensitivity=1.0):
        """
        Args:
          bm_segment (BasilarMembrane.Segment): the BM location this OHC serves
          prestin_gain (float): max active gain in μm displacement per μm input
          inh_sensitivity (float): how strongly efferent ACh inhibits gain
        """
        self.seg = bm_segment
        self.gain = prestin_gain
        self.inh_sens = inh_sensitivity
        self.efferent_level = 0.0  # 0 (no inhibition) to 1 (max inhibition)

    def apply_efferent(self, ach_conc, gaba_conc=0.0):
        """
        Simulate MOC efferent effect: ACh activates α9α10→SK hyperpolarization, GABA co‑release.
        Both reduce OHC gain.
        """
        # ACh effect via SK channel causes hyperpolarization ∝ ach_conc
        ach_effect = self.inh_sens * ach_conc
        # GABA adds extra inhibition
        gaba_effect = 0.5 * gaba_conc
        # combine and clamp
        self.efferent_level = min(1.0, ach_effect + gaba_effect)

    def process(self):
        """
        Enhance or inhibit BM displacement:
        Modified_disp = passive_disp + (1 - efferent_level) * gain * passive_disp.
        """
        passive = self.seg.displacement
        active = (1.0 - self.efferent_level) * self.gain * passive
        # update BM displacement
        self.seg.displacement = passive + active
