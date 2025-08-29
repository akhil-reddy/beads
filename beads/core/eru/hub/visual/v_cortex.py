import numpy as np
from scipy.signal import fftconvolve, convolve2d

from beads.core.eru.interneuron import ShortTermSynapse, MultiCompartmentNeuron, Receptor


# ---------------------------
# Visual STRF (Gabor-like) used by V1 units
# ---------------------------
def make_gabor(k_size, sigma, theta, lam, psi=0, gamma=0.5):
    xs = np.linspace(-k_size // 2, k_size // 2, k_size)
    xv, yv = np.meshgrid(xs, xs)
    x_theta = xv * np.cos(theta) + yv * np.sin(theta)
    y_theta = -xv * np.sin(theta) + yv * np.cos(theta)
    gb = np.exp(-0.5 * (x_theta ** 2 + (gamma ** 2) * y_theta ** 2) / (sigma ** 2)) * np.cos(
        2 * np.pi * x_theta / lam + psi)
    return gb


class VisualSTRF:
    def __init__(self, spatial_size=21, orientations=8, scales=None, temporal_taps=9,
                 temporal_sigma=0.008, modulation_rate=4.0, fs=100.0):
        if scales is None:
            scales = [3, 6]
        self.spatial_size = spatial_size
        self.orientations = orientations
        bank = []
        for sigma in scales:
            for i in range(orientations):
                theta = i * np.pi / orientations
                lam = max(2.0, sigma)
                bank.append(make_gabor(spatial_size, sigma, theta, lam))
        self.bank = np.stack([b / (np.linalg.norm(b) + 1e-12) for b in bank], axis=0)
        self.F = self.bank.shape[0]
        times = np.arange(temporal_taps) / fs
        delay = (temporal_taps // 2) / fs
        gauss = np.exp(-0.5 * ((times - delay) / temporal_sigma) ** 2)
        carrier = np.cos(2 * np.pi * modulation_rate * (times - delay))
        tmp = gauss * carrier
        self.temporal = tmp / (np.linalg.norm(tmp) + 1e-12)

    def apply(self, video_gray):
        # video_gray: (T, H, W)
        T, H, W = video_gray.shape
        F = self.F
        spat_maps = np.zeros((T, F, H, W))
        for t in range(T):
            for f in range(F):
                spat_maps[t, f] = convolve2d(video_gray[t], self.bank[f], mode='same', boundary='symm')
        Tt = len(self.temporal)
        T_out = T - Tt + 1
        out = np.zeros((T_out, F, H, W))
        for f in range(F):
            for i in range(H):
                for j in range(W):
                    out[:, f, i, j] = np.convolve(spat_maps[:, f, i, j], self.temporal, mode='valid')
        return out  # (T_out, F, H, W)


# ---------------------------
# LGN class (M & P streams + center-surround)
# ---------------------------
class LGN:
    def __init__(self, cs_size=15, cs_sigma_c=1.0, cs_sigma_s=3.0, fs=100.0):
        xs = np.arange(cs_size) - cs_size // 2
        xv, yv = np.meshgrid(xs, xs)
        kern = (np.exp(-(xv ** 2 + yv ** 2) / (2 * cs_sigma_c ** 2)) -
                np.exp(-(xv ** 2 + yv ** 2) / (2 * cs_sigma_s ** 2)))
        self.cs_kernel = kern / (np.sum(np.abs(kern)) + 1e-12)
        self.fs = fs
        tM = np.arange(0, 0.05, 1.0 / fs)
        tP = np.arange(0, 0.12, 1.0 / fs)
        self.kernel_M = (np.exp(-tM / 0.01) * (1 - np.exp(-tM / 0.004)))
        self.kernel_M /= (np.linalg.norm(self.kernel_M) + 1e-12)
        self.kernel_P = np.exp(-tP / 0.03)
        self.kernel_P /= (np.linalg.norm(self.kernel_P) + 1e-12)
        self.burst_thresh = 0.2

    def center_surround(self, frame):
        return fftconvolve(frame, self.cs_kernel, mode='same')

    def apply(self, frames):
        # frames shape: (T, H, W) single luminance channel
        T, H, W = frames.shape
        # center-surround
        cs = np.stack([self.center_surround(frames[t]) for t in range(T)])
        # convolve temporal with M and P (valid)
        Tm = len(self.kernel_M)
        Tp = len(self.kernel_P)
        T_out = T - max(Tm, Tp) + 1
        M_out = np.zeros((T_out, H, W))
        P_out = np.zeros((T_out, H, W))
        for i in range(H):
            for j in range(W):
                M_out[:, i, j] = np.convolve(cs[:, i, j], self.kernel_M, mode='valid')
                P_out[:, i, j] = np.convolve(cs[:, i, j], self.kernel_P, mode='valid')[:T_out]
        # burst gating: a simple nonlinearity for M stream
        burst = M_out > self.burst_thresh
        return {'M': M_out, 'P': P_out, 'burst': burst}


# ---------------------------
# V1 class (STRF bank, population of units using MultiCompartmentNeuron)
# ---------------------------
class V1:
    def __init__(self, img_H, img_W, fs=100.0, n_units=64, neuron_params=None, syn_params=None):
        self.H = img_H
        self.W = img_W
        self.fs = fs
        self.strf = VisualSTRF(spatial_size=21, orientations=8, scales=[3, 6], temporal_taps=9, fs=fs)
        self.n_units = n_units
        self.units = []
        if neuron_params is None:
            neuron_params = {
                'C': [200e-12, 200e-12], 'g_L': [10e-9, 10e-9], 'E_L': -65e-3,
                'g_Na': 1200e-9, 'E_Na': 50e-3, 'g_K': 360e-9, 'E_K': -77e-3,
                'g_c': 5e-9, 'dt': 1.0 / fs
            }
        if syn_params is None:
            syn_params = {'U': 0.4, 'tau_rec': 0.4, 'tau_fac': 0.0, 'dt': 1.0 / fs}
        # build grid of units tiled over image, each picks a filter index
        grid = int(np.sqrt(n_units))
        coords = []
        for i in range(grid):
            for j in range(grid):
                ci = int((i + 0.5) * img_H / grid)
                cj = int((j + 0.5) * img_W / grid)
                coords.append((ci, cj))
        for k in range(n_units):
            f_idx = k % self.strf.F
            neu = MultiCompartmentNeuron(neuron_params.copy())
            # add AMPA & NMDA receptors
            neu.add_receptor(Receptor(g_max=1.0e-9, E_rev=0.0, tau_rise=0.0008,
                                      tau_decay=0.004, location='soma', name='AMPA', voltage_dependent=False))
            neu.add_receptor(Receptor(g_max=0.6e-9, E_rev=0.0, tau_rise=0.004,
                                      tau_decay=0.08, location='dend', name='NMDA', voltage_dependent=True))
            syn = ShortTermSynapse(**syn_params)
            unit = {'f_idx': f_idx, 'center': coords[k % len(coords)], 'neu': neu, 'syn': syn}
            self.units.append(unit)

    def process(self, video_gray):
        """
        video_gray: (T, H, W) luminance input
        returns: list of spike-time arrays (seconds) per unit
        """
        feature_maps = self.strf.apply(video_gray)  # (T', F, H, W)
        T_out = feature_maps.shape[0]
        spikes_out = []
        for unit in self.units:
            f = unit['f_idx']
            ci, cj = unit['center']
            # extract local patch (3x3) around center and average across patch
            r = 1
            i0, i1 = max(0, ci - r), min(self.H, ci + r + 1)
            j0, j1 = max(0, cj - r), min(self.W, cj + r + 1)
            patch = feature_maps[:, f, i0:i1, j0:j1]  # (T', h, w)
            drive = patch.reshape(T_out, -1).mean(axis=1)  # (T',)
            # drive -> prespikes (Poisson link)
            drive_pos = np.maximum(drive, 0.0)
            lam = drive_pos / (np.max(drive_pos) + 1e-12) * 30.0
            p = 1.0 - np.exp(-lam * unit['neu'].dt)
            prespikes = (np.random.rand(p.size) < p).astype(float)
            rel = unit['syn'].step(prespikes)
            # map releases to receptors: AMPA weight 1, NMDA 0.6
            syn_rel = np.stack([rel * 1.0, rel * 0.6], axis=1)
            times, Vs = unit['neu'].step(syn_rel)
            # detect peaks as spikes (simple threshold)
            thr = -30e-3
            idx = np.where(Vs >= thr)[0]
            spike_times = np.unique(idx) * unit['neu'].dt  # convert to seconds
            spikes_out.append(spike_times)
        return spikes_out


# ---------------------------
# V2 / V4 / IT (using MultiCompartmentNeuron)
# ---------------------------
def bin_spikes_to_matrix(spike_lists, T_bins, dt):
    """
    Convert a list of spike-time arrays (seconds) to binary matrix (N_units, T_bins).
    dt is time per bin (s).
    """
    N = len(spike_lists)
    mat = np.zeros((N, T_bins), dtype=float)
    for i, times in enumerate(spike_lists):
        if len(times) == 0:
            continue
        # convert times to bin indices
        idx = np.floor(np.asarray(times) / dt).astype(int)
        idx = idx[(idx >= 0) & (idx < T_bins)]
        # allow multi-spikes per bin by counting
        unique, counts = np.unique(idx, return_counts=True)
        mat[i, unique] = counts
    return mat


class V2Unit:
    """
    A V2 neuron that pools a small neighborhood of V1 units with orientation-weighted
    excitatory inputs and an inhibitory surround pool for divisive normalization.
    Integrates presynaptic release via ShortTermSynapse and MultiCompartmentNeuron.
    """

    def __init__(self, presyn_indices, weights, inh_indices=None,
                 neuron_params=None, syn_params=None):
        self.presyn = np.array(presyn_indices, dtype=int)  # indices of V1 units
        self.w = np.array(weights, dtype=float)  # same length
        self.inh = np.array(inh_indices, dtype=int) if inh_indices is not None else np.array([], dtype=int)
        # create synapses (one ShortTermSynapse per excitatory presynapse for realism)
        self.syn_exc = [ShortTermSynapse(**(syn_params or {})) for _ in self.presyn]
        self.syn_inh = [ShortTermSynapse(**(syn_params or {})) for _ in self.inh]
        self.neu = MultiCompartmentNeuron(neuron_params.copy())  # expects receptors to be added by caller
        # caller should add AMPA/NMDA receptors to self.neu
        # scaling constants:
        self.exc_gain = 1.0
        self.inh_gain = 0.8

    def run(self, presyn_spike_mat, T_bins):
        """
        presyn_spike_mat: (N_v1, T_bins) binary/count matrix
        Returns: spike times array (seconds) from this V2 unit
        """
        # build per-bin excitatory drive: weighted sum of presyn spike counts
        drives = np.zeros(T_bins, dtype=float)
        for k, idx in enumerate(self.presyn):
            s = presyn_spike_mat[idx]  # (T_bins,)
            rel = self.syn_exc[k].step(s)  # release per-bin
            drives += self.w[k] * rel
        # inhibitory divisive normalization using summed inh activity
        inh_drive = 0.0
        if len(self.inh) > 0:
            inh_sum = np.zeros(T_bins, dtype=float)
            for k, idx in enumerate(self.inh):
                s = presyn_spike_mat[idx]
                rel = self.syn_inh[k].step(s)
                inh_sum += rel
            # implement divisive normalization by scaling excitation
            denom = 1.0 + self.inh_gain * np.convolve(inh_sum, np.ones(3) / 3.0, mode='same')
            drives /= denom + 1e-12

        # Map drives to receptors and step neuron
        # Map drives -> [AMPA, NMDA] release amplitudes (simple split)
        syn_release = np.stack([drives * self.exc_gain, drives * (0.5 * self.exc_gain)], axis=1)
        times, Vs = self.neu.step(syn_release, t0=0.0)
        # simple spike detection on Vs peaks
        thr = -30e-3
        idx = np.where(Vs >= thr)[0]
        spike_bins = np.unique(idx)
        spike_times = spike_bins * self.neu.dt
        return spike_times


class V2:
    """
    V2 layer that constructs units by pooling localized V1 indices.
    """

    def __init__(self, v1_obj, n_units=64, pool_radius=2, neuron_params=None, syn_params=None):
        self.v1 = v1_obj
        self.n_units = n_units
        self.pool_radius = pool_radius
        # infer grid layout of v1 from v1 unit centers if present, else assume square
        Nv1 = len(self.v1.units)
        grid = int(np.sqrt(Nv1))
        coords = np.array([u['center'] for u in self.v1.units]) if 'center' in self.v1.units[0] else \
            np.array([(i, j) for i in range(grid) for j in range(grid)])
        # build list of indices per unit
        self.units = []
        rng = np.random.default_rng(0)
        for n in range(n_units):
            # pick center at random across V1 units
            center_idx = rng.integers(0, Nv1)
            ci, cj = coords[center_idx]
            # select neighbors within euclidean radius in pixel-space by comparing centers
            dists = np.linalg.norm(coords - coords[center_idx], axis=1)
            presyn_idx = np.where(dists <= pool_radius)[0].tolist()
            # weights: gaussian by distance
            sigma = max(1.0, pool_radius / 2.0)
            weights = np.exp(-0.5 * (dists[presyn_idx] / sigma) ** 2)
            # inhibitory pool: choose some units at slightly larger radius
            inh_idx = np.where((dists > pool_radius) & (dists <= pool_radius * 2))[0].tolist()
            # create V2 unit
            unit = V2Unit(presyn_idx, weights, inh_indices=inh_idx,
                          neuron_params=(neuron_params or self._default_neuron_params()),
                          syn_params=(syn_params or self._default_syn_params()))
            # add receptors (AMPA soma, NMDA dend) to unit's neuron
            unit.neu.add_receptor(
                Receptor(g_max=1.0e-9, E_rev=0.0, tau_rise=0.0008, tau_decay=0.004, location='soma', name='AMPA',
                         voltage_dependent=False))
            unit.neu.add_receptor(
                Receptor(g_max=0.6e-9, E_rev=0.0, tau_rise=0.004, tau_decay=0.08, location='dend', name='NMDA',
                         voltage_dependent=True))
            self.units.append(unit)

    def _default_neuron_params(self):
        return {'C': [200e-12, 200e-12], 'g_L': [10e-9, 10e-9], 'E_L': -65e-3,
                'g_Na': 1200e-9, 'E_Na': 50e-3, 'g_K': 360e-9, 'E_K': -77e-3,
                'g_c': 5e-9, 'dt': 1.0 / self.v1.fs}

    def _default_syn_params(self):
        return {'U': 0.4, 'tau_rec': 0.4, 'tau_fac': 0.0, 'dt': 1.0 / self.v1.fs}

    def process(self, video_gray):
        # compute v1 spikes and bin them
        v1_spike_lists = self.v1.process(video_gray)  # list length Nv1, times in seconds
        # determine T_bins from V1 feature maps length
        T_frames = video_gray.shape[0]
        T_bins = T_frames - len(self.v1.strf.temporal) + 1
        dt = 1.0 / self.v1.fs
        v1_mat = bin_spikes_to_matrix(v1_spike_lists, T_bins, dt)  # (Nv1, T_bins)

        v2_out = []
        for unit in self.units:
            st = unit.run(v1_mat, T_bins)
            v2_out.append(st)
        return v2_out


# ---------------------------
# V4: curvature + color pooling (more global; multiplicative combos)
# ---------------------------
class V4Unit:
    """
    Pools multiple V2 units with learned weights; can detect curvature by combining oriented patches.
    """

    def __init__(self, presyn_indices, weights, neuron_params=None, syn_params=None):
        self.presyn = np.array(presyn_indices, dtype=int)
        self.w = np.array(weights, dtype=float)
        self.syn = [ShortTermSynapse(**(syn_params or {})) for _ in self.presyn]
        self.neu = MultiCompartmentNeuron(neuron_params.copy())

        # add receptors (excitatory AMPA+NMDA)
        self.neu.add_receptor(
            Receptor(g_max=1.2e-9, E_rev=0.0, tau_rise=0.0008, tau_decay=0.004, location='soma', name='AMPA'))
        self.neu.add_receptor(
            Receptor(g_max=0.8e-9, E_rev=0.0, tau_rise=0.004, tau_decay=0.08, location='dend', name='NMDA',
                     voltage_dependent=True))

    def run(self, presyn_spike_mat, T_bins):
        # compute multiplicative pairwise features for curvature:
        # strategy: compute linear drive, plus second-order drive from pairs of presyn inputs
        linear = np.zeros(T_bins, dtype=float)
        for k, idx in enumerate(self.presyn):
            rel = self.syn[k].step(presyn_spike_mat[idx])
            linear += self.w[k] * rel
        # second order: take pairwise products for nearby presyn indices (approx curvature)
        second = np.zeros(T_bins, dtype=float)
        for a in range(len(self.presyn) - 1):
            rel_a = self.syn[a].step(presyn_spike_mat[self.presyn[a]])
            rel_b = self.syn[a + 1].step(presyn_spike_mat[self.presyn[a + 1]])
            second += 0.5 * (rel_a * rel_b)
        drive = linear + 0.5 * second
        syn_rel = np.stack([drive, 0.6 * drive], axis=1)
        times, Vs = self.neu.step(syn_rel)
        thr = -30e-3
        idx = np.where(Vs >= thr)[0]
        return np.unique(idx) * self.neu.dt


class V4:
    def __init__(self, v2_obj, n_units=32):
        self.v2 = v2_obj
        self.n_units = n_units
        Nv2 = len(self.v2.units) if hasattr(self.v2, 'units') else len(
            self.v2.process(np.zeros((1, self.v2.v1.H, self.v2.v1.W))))
        rng = np.random.default_rng(1)
        self.units = []
        # each V4 unit pools ~K random V2 units covering object parts (can be made spatially structured)
        K = min(12, max(4, Nv2 // 8))
        for _ in range(n_units):
            pres = rng.choice(range(Nv2), size=K, replace=False).tolist()
            weights = rng.normal(1.0, 0.2, size=K)
            unit = V4Unit(pres, weights, neuron_params=self.v2._default_neuron_params(),
                          syn_params=self.v2._default_syn_params())
            self.units.append(unit)

    def process(self, video_gray):
        v2_spike_lists = self.v2.process(video_gray)
        T_bins = video_gray.shape[0] - len(self.v2.v1.strf.temporal) + 1
        dt = 1.0 / self.v2.v1.fs
        v2_mat = bin_spikes_to_matrix(v2_spike_lists, T_bins, dt)
        out = []
        for unit in self.units:
            out.append(unit.run(v2_mat, T_bins))
        return out


# ---------------------------
# IT: associative readout using MultiCompNeuron population
# ---------------------------
class ITUnit:
    def __init__(self, presyn_indices, weights, neuron_params=None, syn_params=None):
        self.presyn = np.array(presyn_indices, dtype=int)
        self.w = np.array(weights, dtype=float)
        self.syn = [ShortTermSynapse(**(syn_params or {})) for _ in self.presyn]
        self.neu = MultiCompartmentNeuron(neuron_params.copy())
        self.neu.add_receptor(
            Receptor(g_max=1.5e-9, E_rev=0.0, tau_rise=0.0008, tau_decay=0.004, location='soma', name='AMPA'))
        self.neu.add_receptor(
            Receptor(g_max=1.0e-9, E_rev=0.0, tau_rise=0.004, tau_decay=0.08, location='dend', name='NMDA',
                     voltage_dependent=True))

    def run(self, presyn_mat, T_bins):
        drive = np.zeros(T_bins, dtype=float)
        for k, idx in enumerate(self.presyn):
            rel = self.syn[k].step(presyn_mat[idx])
            drive += self.w[k] * rel
        syn_rel = np.stack([drive, 0.6 * drive], axis=1)
        times, Vs = self.neu.step(syn_rel)
        thr = -30e-3
        idx = np.where(Vs >= thr)[0]
        return np.unique(idx) * self.neu.dt


class IT:
    def __init__(self, v4_obj, n_units=32):
        self.v4 = v4_obj
        self.n_units = n_units
        Nv4 = len(self.v4.units)
        rng = np.random.default_rng(2)
        self.units = []
        K = min(20, max(6, Nv4 // 4))
        for _ in range(n_units):
            pres = rng.choice(range(Nv4), size=K, replace=False).tolist()
            w = rng.normal(1.0, 0.3, size=K)
            unit = ITUnit(pres, w, neuron_params=self.v4.v2._default_neuron_params(),
                          syn_params=self.v4.v2._default_syn_params())
            self.units.append(unit)

    def process(self, video_gray):
        v4_spike_lists = self.v4.process(video_gray)
        T_bins = video_gray.shape[0] - len(self.v4.v2.v1.strf.temporal) + 1
        dt = 1.0 / self.v4.v2.v1.fs
        v4_mat = bin_spikes_to_matrix(v4_spike_lists, T_bins, dt)
        out = []
        for unit in self.units:
            out.append(unit.run(v4_mat, T_bins))
        return out


# ----------------------------------------
# 5. Visual Cortex Full Integration
# ----------------------------------------
class VisualCortex:
    def __init__(self, T=60, H=64, W=64):
        frames = np.clip(np.random.randn(T, H, W) * 0.05 + 0.5, 0.0, 1.0)
        # use luminance only for LGN/V1 pipeline
        self.luminance = frames  # assume already single-channel for simplicity

        self.lgn = LGN(cs_size=15, fs=100.0)
        self.v1 = V1(img_H=H, img_W=W, fs=100.0, n_units=36)
        self.v2 = V2(self.v1)
        self.v4 = V4(self.v2)
        self.it = IT(self.v4)

    def run(self):
        lgn_out = self.lgn.apply(self.luminance)
        v1_spikes = self.v1.process(self.luminance)
        v2_spikes = self.v2.process(self.luminance)
        v4_spikes = self.v4.process(self.luminance)
        it_spikes = self.it.process(self.luminance)

        print("V1 spikes per unit (counts):", [len(s) for s in v1_spikes[:8]])
        print("IT unit spike example counts:", [len(s) for s in it_spikes[:8]])
