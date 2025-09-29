from typing import Union, Dict, Any, Sequence, Optional
from scipy.signal import butter, sosfilt
import numpy as np


class BasilarMembrane:
    class Segment:
        def __init__(self, cf, width, stiffness, mass):
            self.cf = float(cf)
            self.width = float(width)
            self.stiffness = float(stiffness)
            self.mass = float(mass)
            self.velocity = 0.0
            self.displacement = 0.0
            self.force = 0.0

    def __init__(self, n=200, cf_range=(20, 20000), duct_depth: float = 5e-4):
        """
        n: number of discrete BM segments
        cf_range: (low_cf, high_cf) in Hz (will be logged-space assigned along the array)
        duct_depth: effective depth of the cochlear duct used to convert pressure (Pa) -> force (N)
        """
        # create centre frequencies from high -> low (matching earlier code)
        cfs = np.logspace(np.log10(cf_range[1]), np.log10(cf_range[0]), n)
        widths = np.linspace(8e-5, 6.5e-4, n)  # meters
        stiffness = 1.0 / widths
        mass = np.full(n, 1e-9)
        self.segments = [BasilarMembrane.Segment(cf, w, k, m)
                         for cf, w, k, m in zip(cfs, widths, stiffness, mass)]

        # store arrays for vectorized ops & convenience
        self.n = n
        self.cfs = np.asarray(cfs, dtype=float)
        self.widths = np.asarray(widths, dtype=float)
        self.stiffness = np.asarray(stiffness, dtype=float)
        self.masses = np.asarray(mass, dtype=float)

        # duct_depth used to convert pressure (Pa) -> force: force = pressure * (width * duct_depth)
        self.duct_depth = float(duct_depth)

    def apply_pressure(self, pressure: Union[Dict[Any, float], Sequence[float], np.ndarray]):
        """
        Accepts either:
          - a dict mapping CF -> pressure (Pa) (keys can be floats or ints),
          - or a 1D array-like of length == n (pressure per BM segment in Pa)
        The dict form will be interpolated over the BM CF positions if needed.
        After conversion to a per-segment array p_seg (length n), seg.force += p_seg * area.
        """
        # convert dict -> per-segment pressure array by interpolation
        if isinstance(pressure, dict):
            # Expect keys numeric (CFs) and values numeric (Pa). Build arrays and interpolate.
            keys = np.array(sorted([float(k) for k in pressure.keys()]), dtype=float)
            vals = np.array([float(pressure[k]) for k in sorted(pressure.keys())], dtype=float)
            if keys.size == 0:
                p_seg = np.zeros(self.n, dtype=float)
            elif keys.size == 1:
                # single key -> constant pressure across all segments
                p_seg = np.full(self.n, vals[0], dtype=float)
            else:
                # interpolate in log-frequency domain to be natural for cochlea
                # convert both to log10 to interpolate by log-frequency
                log_keys = np.log10(keys)
                log_cfs = np.log10(self.cfs)
                # handle possible out-of-bounds by np.interp's left/right args
                log_vals = np.interp(log_cfs, log_keys, vals)
                p_seg = log_vals
        else:
            # expect array-like
            arr = np.asarray(pressure, dtype=float)
            if arr.ndim != 1:
                raise ValueError("pressure must be 1D array-like or dict mapping CF->pressure")
            if arr.size == self.n:
                p_seg = arr
            elif arr.size == 1:
                p_seg = np.full(self.n, arr.item(), dtype=float)
            else:
                # If lengths mismatch, try to interpolate along index
                # map provided array indices to BM indices linearly
                x = np.linspace(0, 1, arr.size)
                xi = np.linspace(0, 1, self.n)
                p_seg = np.interp(xi, x, arr)

        # Convert pressure (Pa) -> force (N) using effective area = width * duct_depth
        areas = self.widths * self.duct_depth  # (n,)
        forces = p_seg * areas  # per-segment forces in Newtons

        # add forces to segment force accumulators (vectorized)
        for seg, f in zip(self.segments, forces):
            seg.force += float(f)

    def function(self, dt: float, damping: Optional[float] = None, coupling: Optional[float] = None):
        """
        Step BM dynamics for a time-step dt (seconds).

        dt: time step (s)
        damping: optional viscous damping coefficient b (N s / m). If None, no damping applied.
        coupling: optional coupling coefficient gamma used in discrete Laplacian
                  (coupling*(x_{i+1} - 2 x_i + x_{i-1})).
        """
        # defaults
        if damping is None:
            b = 0.0
        else:
            b = float(damping)
        gamma = 0.0 if coupling is None else float(coupling)

        # gather arrays for vectorized update
        x = np.array([s.displacement for s in self.segments], dtype=float)
        v = np.array([s.velocity for s in self.segments], dtype=float)
        f = np.array([s.force for s in self.segments], dtype=float)
        k = self.stiffness
        m = self.masses

        # compute coupling term (discrete Laplacian) if requested
        if gamma != 0.0:
            lap = np.zeros_like(x)
            lap[1:-1] = x[2:] - 2 * x[1:-1] + x[:-2]
        else:
            lap = np.zeros_like(x)

        # acceleration
        acc = (f - k * x - b * v + gamma * lap) / m

        # semi-implicit Euler (more stable for oscillators):
        v = v + acc * dt
        x = x + v * dt

        # store back
        for i, seg in enumerate(self.segments):
            seg.velocity = float(v[i])
            seg.displacement = float(x[i])
            seg.force = 0.0  # reset after applying


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
        Veff = self.Vm - (self.cl_sens * (self.Cl - 140) / 1000.0)  # Cl shifts V half
        return 1.0 / (1 + np.exp(-(Veff - self.V_half) / self.z_slope))

    def electromechanical_force(self):
        # Force ~ stiffness × OHC length change
        frac = self.prestin_fraction() * (1 - self.eff * self.inh_gain)
        dz = (frac * 2 - 1) * self.z_max  # length change ±z_max
        return self.seg.stiffness * dz

    def function(self, dt, synaptic_input):
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

    def function(self, rate):
        # rate: 0–100% release → sets efferent level on each OHC
        for ohc in self.ohcs:
            ohc.eff = np.clip(rate, 0.0, 1.0)


def make_gammatone_bank(cfs, fs, bw_frac=1.0):
    """
    Returns second-order-sections bandpass filters approximating gammatone bands.
    Simpler: use Butterworth bandpasses centered at CF with bandwidth ~ ERB.
    """
    sos_list = []
    for cf in cfs:
        # approximate ERB: ERB(cf) = 24.7*(4.37*cf/1000 + 1)
        erb = 24.7 * (4.37 * cf / 1000.0 + 1.0) * bw_frac
        low = max(1.0, cf - erb / 2.0)
        high = min(fs / 2 - 1, cf + erb / 2.0)
        if low >= high:
            # fallback narrowband
            low = max(0.1, cf * 0.9)
            high = min(fs / 2.0 - 1, cf * 1.1)
        sos = butter(2, [low / (fs / 2), high / (fs / 2)], btype='bandpass', output='sos')
        sos_list.append(sos)
    return sos_list


def bank_filter(s, sos_list):
    # s: 1D waveform
    outs = []
    for sos in sos_list:
        outs.append(sosfilt(sos, s))
    return np.stack(outs, axis=0)  # shape (n_bands, T)


def run(waveform, fs, bm: BasilarMembrane, ohcs: list, moc: MedialOlivocochlear, chunk_dt=1.0 / 40000.0):
    """
    waveform: 1D numpy array (sound pressure in Pa at ear canal sampling rate = fs)
    fs: sample rate of waveform (Hz)
    bm: BasilarMembrane instance
    ohcs: list of OuterHairCell instances matched to bm.segments
    moc: MedialOlivocochlear instance

    chunk_dt: integration dt for BM/OHC (seconds). Ideally <= 1/fs. We'll resample/filter accordingly.
    """
    # 1) prepare filterbank: use one band per BM segment's CF (or downsample BM segments)
    seg_cfs = np.array([seg.cf for seg in bm.segments])
    n_segments = len(seg_cfs)
    # for efficiency you can sample a subset of segments; here we match counts
    sos_list = make_gammatone_bank(seg_cfs, fs)
    # 2) filter whole waveform into band signals (shape n_segments x T)
    band_outputs = bank_filter(waveform, sos_list)  # (n_segments, T_in)

    # 3) decide integration dt and number of input samples per dt bin
    dt = chunk_dt
    samples_per_dt = int(round(dt * fs))
    if samples_per_dt < 1:
        samples_per_dt = 1
        dt = 1.0 / fs  # use input sample period
    T_in = waveform.shape[0]
    n_steps = T_in // samples_per_dt

    # effective area per segment (width * depth). tune depth (m)
    duct_depth = 5e-4  # 0.5 mm; tune
    seg_areas = np.array([seg.width * duct_depth for seg in bm.segments])  # (n_segments,)

    # step through time bins
    for step in range(n_steps):
        i0 = step * samples_per_dt
        i1 = i0 + samples_per_dt
        # average band output over bin -> approximate pressure per band (Pa)
        p_seg = band_outputs[:, i0:i1].mean(axis=1)  # (n_segments,)

        pressure_map = {cf: float(p) for cf, p in zip(seg_cfs, p_seg)}
        bm.apply_pressure(pressure_map)  # adds F via seg.force += p*width in your code

        # advance BM dynamics by dt
        bm.function(dt)

        # compute MET input for OHCs (simple conversion: BM displacement -> deflection -> current)
        # For simplicity: I_met = g_max * sigmoid( k * displacement )

        g_max = 200e-12  # 200 pA max current (tune)
        k = 1e5  # slope converting meters->unitless (tune)
        x0 = 0.0
        # collect segment displacements
        disps = np.array([seg.displacement for seg in bm.segments])
        popen = 1.0 / (1.0 + np.exp(-k * (disps - x0)))  # (n_segments,)
        I_met = g_max * popen  # (A) per hair cell

        # step OHCs and also an IHC (not shown) — assuming ohcs list aligns 1:1 with bm.segments
        for j, ohc in enumerate(ohcs):
            syn_current = I_met[j]
            ohc.function(dt, syn_current)

    # end loop
