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


# ---------- Helper: convert spike-time lists -> binned matrix ----------
def bin_spikes_to_matrix(spike_lists, T_bins, dt):
    N = len(spike_lists)
    mat = np.zeros((N, T_bins), dtype=float)
    for i, times in enumerate(spike_lists):
        if len(times) == 0:
            continue
        idx = np.floor(np.asarray(times) / dt).astype(int)
        idx = idx[(idx >= 0) & (idx < T_bins)]
        if idx.size == 0:
            continue
        unique, counts = np.unique(idx, return_counts=True)
        mat[i, unique] = counts
    return mat


# ---------- LGN (tonic/burst + M/P stream separation) ----------
class LGN:
    """
    LGN: center-surround + temporal M/P kernels.
    Maintains a short activity history to switch M cells between tonic and burst modes.
    - get_activity_history(frames) computes running RMS of inputs; low recent activity -> burst mode.
    - apply(frames) returns {'M','P','burst_gain'} where burst_gain is multiplicative factor per (t,i,j)
    """

    def __init__(self, cs_size=15, cs_sigma_c=1.0, cs_sigma_s=3.0, fs=100.0, history_len=40, burst_gain=2.0):
        xs = np.arange(cs_size) - cs_size // 2
        xv, yv = np.meshgrid(xs, xs)
        kern = (np.exp(-(xv ** 2 + yv ** 2) / (2 * cs_sigma_c ** 2)) - np.exp(
            -(xv ** 2 + yv ** 2) / (2 * cs_sigma_s ** 2)))
        self.cs_kernel = kern / (np.sum(np.abs(kern)) + 1e-12)
        self.fs = fs
        tM = np.arange(0, 0.05, 1.0 / fs)
        tP = np.arange(0, 0.12, 1.0 / fs)
        self.kernel_M = (np.exp(-tM / 0.01) * (1 - np.exp(-tM / 0.004)))
        self.kernel_M /= (np.linalg.norm(self.kernel_M) + 1e-12)
        self.kernel_P = np.exp(-tP / 0.03)
        self.kernel_P /= (np.linalg.norm(self.kernel_P) + 1e-12)
        self.history_len = history_len
        self.burst_gain_scalar = burst_gain
        # keep circular buffer for recent CS energy
        self.history_buffer = None

    def center_surround(self, frame):
        return fftconvolve(frame, self.cs_kernel, mode='same')

    def _update_history(self, cs):
        # cs: (T, H, W)
        energy = np.sqrt(np.mean(cs ** 2, axis=(1, 2)))  # per-frame RMS energy
        if self.history_buffer is None:
            self.history_buffer = energy[-self.history_len:].tolist() if len(
                energy) >= self.history_len else energy.tolist()
        else:
            # append new frames to buffer, truncate
            self.history_buffer.extend(energy.tolist())
            self.history_buffer = self.history_buffer[-self.history_len:]

    def compute_burst_mask(self):
        """
        Decide burst mode based on recent activity: if mean activity in buffer below threshold -> burst.
        Return scalar burst_factor in (1.0, burst_gain_scalar)
        """
        if self.history_buffer is None or len(self.history_buffer) == 0:
            return 1.0
        recent = np.array(self.history_buffer)
        mean_act = np.mean(recent)
        # biologically: silence / low drive predisposes T-type Ca2+ deinactivation -> bursts.
        # here threshold is arbitrarily set relative to buffer variance; tune empirically.
        thresh = 0.5 * (np.max(recent) + np.min(recent) + 1e-12)
        if mean_act < thresh:
            return self.burst_gain_scalar
        return 1.0

    def apply(self, frames):
        """
        frames: (T, H, W) luminance
        returns dict with M_out (T',H,W), P_out (T',H,W), burst_factor (scalar)
        """
        T, H, W = frames.shape
        cs = np.stack([self.center_surround(frames[t]) for t in range(T)])  # (T,H,W)
        self._update_history(cs)
        burst_gain = self.compute_burst_mask()
        Tm = len(self.kernel_M)
        Tp = len(self.kernel_P)
        T_out = T - max(Tm, Tp) + 1
        M_out = np.zeros((T_out, H, W))
        P_out = np.zeros((T_out, H, W))
        for i in range(H):
            for j in range(W):
                M_out[:, i, j] = np.convolve(cs[:, i, j], self.kernel_M, mode='valid')
                P_out[:, i, j] = np.convolve(cs[:, i, j], self.kernel_P, mode='valid')[:T_out]
        # apply burst gain multiplicatively to the M stream when M_out is transiently suprathreshold
        # but return as scalar gain to be applied by downstream L4 units (for performance)
        return {'M': M_out, 'P': P_out, 'burst_gain': burst_gain, 'cs': cs}


# ---------- V1 with L4, L2/3 and L5/6 ----------

class L4Unit:
    """
    Layer 4 simple cell: picks a filter index f_idx and a spatial center,
    computes drive from VisualSTRF feature maps and outputs spike-times via neuron+synapse.
    """

    def __init__(self, f_idx, center, neuron_params, syn_params, pool_radius=1, amp=30.0):
        self.f_idx = f_idx
        self.center = center  # (i,j) pixel center
        self.neu = MultiCompartmentNeuron(neuron_params.copy())
        # add AMPA & NMDA receptors
        self.neu.add_receptor(
            Receptor(g_max=1.0e-9, E_rev=0.0, tau_rise=0.0008, tau_decay=0.004, location='soma', name='AMPA'))
        self.neu.add_receptor(
            Receptor(g_max=0.6e-9, E_rev=0.0, tau_rise=0.004, tau_decay=0.08, location='dend', name='NMDA',
                     voltage_dependent=True))
        self.syn = ShortTermSynapse(**syn_params)
        self.pool_radius = pool_radius
        self.amp = amp  # scaling to convert drive -> firing rate

    def receptive_drive(self, feature_maps):
        # feature_maps: (T, F, H, W)
        T, F, H, W = feature_maps.shape
        i, j = self.center
        r = self.pool_radius
        i0, i1 = max(0, i - r), min(H, i + r + 1)
        j0, j1 = max(0, j - r), min(W, j + r + 1)
        patch = feature_maps[:, self.f_idx, i0:i1, j0:j1]
        drive = np.maximum(patch.reshape(T, -1).mean(axis=1), 0.0)
        return drive

    def step(self, feature_maps, burst_gain=1.0, modulation=1.0):
        # modulation is L5/6 feedback multiplicative factor (affects firing gain)
        drive = self.receptive_drive(feature_maps)  # (T,)
        # firing rate scaling -> prob per frame
        lam = (drive / (np.max(drive) + 1e-12)) * self.amp * modulation * burst_gain
        p = 1.0 - np.exp(-lam * self.neu.dt)
        prespikes = (np.random.rand(p.size) < p).astype(float)
        rel = self.syn.step(prespikes)
        syn_rel = np.stack([rel * 1.0, rel * 0.6], axis=1)
        times, Vs = self.neu.step(syn_rel)
        # detect spikes from somatic trace
        thr = -30e-3
        idx = np.where(Vs >= thr)[0]
        return np.unique(idx) * self.neu.dt  # seconds


class L23Unit:
    """
    Pools several L4 units that share similar orientation to stitch collinear edges.
    Implements horizontal facilitation: if neighbors are coactive, increase their short-term 'u' briefly.
    """

    def __init__(self, presyn_indices, neuron_params, syn_params, v1_reference, facilitation_scale=1.2):
        self.presyn = np.array(presyn_indices, dtype=int)
        self.syns = [ShortTermSynapse(**syn_params) for _ in self.presyn]
        self.neu = MultiCompartmentNeuron(neuron_params.copy())
        self.neu.add_receptor(
            Receptor(g_max=1.0e-9, E_rev=0.0, tau_rise=0.0008, tau_decay=0.004, location='soma', name='AMPA'))
        self.neu.add_receptor(
            Receptor(g_max=0.7e-9, E_rev=0.0, tau_rise=0.004, tau_decay=0.08, location='dend', name='NMDA',
                     voltage_dependent=True))
        self.v1 = v1_reference
        self.facilitation_scale = facilitation_scale

    def run(self, l4_spike_lists, T_bins):
        # l4_spike_lists: list of spike arrays (seconds), map to binned counts
        dt = 1.0 / self.v1.fs
        l4_mat = bin_spikes_to_matrix(l4_spike_lists, T_bins, dt)  # (N_l4, T_bins)
        drive = np.zeros(T_bins, dtype=float)
        # horizontal facilitation: if more than one presyn fires at same bin, transiently increase 'u' so more release
        for k, idx in enumerate(self.presyn):
            s = l4_mat[idx]
            # compute local coincidence
            # We'll approximate coincidence by counting presyn activity across all presyn indices
            local_sum = np.sum(l4_mat[self.presyn], axis=0)
            # transient factor
            fac = 1.0 + (self.facilitation_scale - 1.0) * (local_sum > 1.0).astype(float)
            rel = self.syns[k].step(s * fac)
            drive += rel
        syn_rel = np.stack([drive, 0.6 * drive], axis=1)
        times, Vs = self.neu.step(syn_rel)
        thr = -30e-3
        idx = np.where(Vs >= thr)[0]
        return np.unique(idx) * self.neu.dt


class L56:
    """
    Layer 5/6: collects population activity from L2/3 and computes a feedback modulation map
    that can multiplicatively adjust L4 gain (simulating corticothalamic feedback tuning).
    """

    def __init__(self, v1_reference, feedback_strength=0.5):
        self.v1 = v1_reference
        self.feedback_strength = feedback_strength

    def compute_feedback(self, l23_spike_lists, video_shape):
        """
        Simple map: compute average L2/3 activity per spatial tile, produce modulation factors in (0.5,1.5)
        video_shape: (H,W) used to tile L2/3 activity back to pixel-space.
        """
        T_frames = len(l23_spike_lists[0]) if len(l23_spike_lists) and isinstance(l23_spike_lists[0], np.ndarray) else 1
        # coarse approach: compute mean rates for each L2/3 unit and tile to image
        rates = np.array([len(s) for s in l23_spike_lists], dtype=float)
        if np.max(rates) == 0:
            return np.ones(video_shape, dtype=float)
        norm = rates / (np.max(rates) + 1e-12)
        # map L2/3 units to image patches (assume same ordering as v1 grid mapping)
        n_units = len(self.v1.units)
        grid = int(np.sqrt(n_units))
        H, W = video_shape
        tile_H = max(1, H // grid)
        tile_W = max(1, W // grid)
        mod_map = np.ones((H, W), dtype=float)
        k = 0
        for i in range(grid):
            for j in range(grid):
                val = 1.0 + self.feedback_strength * (norm[k] - 0.5)
                i0, i1 = i * tile_H, min(H, (i + 1) * tile_H)
                j0, j1 = j * tile_W, min(W, (j + 1) * tile_W)
                mod_map[i0:i1, j0:j1] = val
                k += 1
        return mod_map


# ---------- V2: corner & border-ownership detectors ----------
class V2UnitCorner:
    """
    Detect simple corners by combining two L4 units oriented ~90deg at nearby spatial offset.
    """

    def __init__(self, idx_a, idx_b, neuron_params, syn_params, v1_ref):
        self.idx_a = idx_a
        self.idx_b = idx_b
        self.syn_a = ShortTermSynapse(**syn_params)
        self.syn_b = ShortTermSynapse(**syn_params)
        self.neu = MultiCompartmentNeuron(neuron_params.copy())
        self.neu.add_receptor(Receptor(g_max=1.0e-9, E_rev=0.0, tau_rise=0.0008, tau_decay=0.004, location='soma'))
        self.neu.add_receptor(
            Receptor(g_max=0.7e-9, E_rev=0.0, tau_rise=0.004, tau_decay=0.08, location='dend', voltage_dependent=True))
        self.v1 = v1_ref

    def run(self, l4_spike_lists, T_bins):
        dt = 1.0 / self.v1.fs
        a = bin_spikes_to_matrix([l4_spike_lists[self.idx_a]], T_bins, dt)[0]
        b = bin_spikes_to_matrix([l4_spike_lists[self.idx_b]], T_bins, dt)[0]
        rel_a = self.syn_a.step(a)
        rel_b = self.syn_b.step(b)
        # coincidence detector: product gives strong output when simultaneous
        drive = rel_a * rel_b
        syn_rel = np.stack([drive, 0.6 * drive], axis=1)
        times, Vs = self.neu.step(syn_rel)
        thr = -30e-3
        idx = np.where(Vs >= thr)[0]
        return np.unique(idx) * self.neu.dt


class V2:
    def __init__(self, v1_obj, n_units=48):
        self.v1 = v1_obj
        self.units = []
        Nv1 = len(self.v1.units)
        rng = np.random.default_rng(7)
        # build corner detectors by choosing nearby pairs with differing filter indices
        for _ in range(n_units):
            a = rng.integers(0, Nv1)
            b = (a + rng.integers(1, max(2, Nv1 // 8))) % Nv1
            # ensure some orientation difference in filters (approx by f_idx)
            unit = V2UnitCorner(a, b, neuron_params=self._default_neuron_params(),
                                syn_params=self._default_syn_params(), v1_ref=self.v1)
            self.units.append(unit)

    def _default_neuron_params(self):
        return {'C': [200e-12, 200e-12], 'g_L': [10e-9, 10e-9], 'E_L': -65e-3,
                'g_Na': 1200e-9, 'E_Na': 50e-3, 'g_K': 360e-9, 'E_K': -77e-3,
                'g_c': 5e-9, 'dt': 1.0 / self.v1.fs}

    def _default_syn_params(self):
        return {'U': 0.4, 'tau_rec': 0.4, 'tau_fac': 0.0, 'dt': 1.0 / self.v1.fs}

    def process(self, video_gray):
        l4_spikes = self.v1.process(video_gray)
        T_bins = video_gray.shape[0] - len(self.v1.strf.temporal) + 1
        out = []
        for u in self.units:
            out.append(u.run(l4_spikes, T_bins))
        return out


# ---------- V4: curvature + color-invariant population pooling ----------
class V4:
    def __init__(self, v2_obj, n_units=32):
        self.v2 = v2_obj
        self.n_units = n_units
        Nv2 = len(self.v2.units)
        rng = np.random.default_rng(11)
        self.units = []
        for _ in range(n_units):
            K = min(12, Nv2)
            pres = rng.choice(range(Nv2), size=K, replace=False).tolist()
            w = rng.normal(1.0, 0.25, size=K)
            u = {'pres': pres, 'w': w,
                 'neu': MultiCompartmentNeuron(self.v2._default_neuron_params())}
            # add receptors
            u['neu'].add_receptor(Receptor(g_max=1.2e-9, E_rev=0.0, tau_rise=0.0008, tau_decay=0.004, location='soma'))
            u['neu'].add_receptor(Receptor(g_max=0.8e-9, E_rev=0.0, tau_rise=0.004, tau_decay=0.08, location='dend',
                                           voltage_dependent=True))
            u['syns'] = [ShortTermSynapse(**self.v2._default_syn_params()) for _ in pres]
            self.units.append(u)

    def process(self, video_gray):
        v2_spikes = self.v2.process(video_gray)
        T_bins = video_gray.shape[0] - len(self.v2.v1.strf.temporal) + 1
        dt = 1.0 / self.v2.v1.fs
        v2_mat = bin_spikes_to_matrix(v2_spikes, T_bins, dt)
        out = []
        for u in self.units:
            drive = np.zeros(T_bins, dtype=float)
            for k, idx in enumerate(u['pres']):
                rel = u['syns'][k].step(v2_mat[idx])
                drive += u['w'][k] * rel
            # lighting invariance: divide by local RMS luminance proxy (here use mean of drive across time)
            denom = np.mean(drive) + 1e-12
            drive_norm = drive / denom
            syn_rel = np.stack([drive_norm, 0.6 * drive_norm], axis=1)
            times, Vs = u['neu'].step(syn_rel)
            thr = -30e-3
            idx_sp = np.where(Vs >= thr)[0]
            out.append(np.unique(idx_sp) * u['neu'].dt)
        return out


# ---------- IT: associative memory / population readout ----------
class IT:
    """
    IT implements a prototype-based associative memory:
    - During `store_prototypes(prototypes)` it stores population activation vectors (V4 activations).
    - During `process`, it computes similarity to stored prototypes and fires IT units that match.
    This is a convenient, biologically-plausible readout: prototypes can be learned via Hebbian rules.
    """

    def __init__(self, v4_obj, n_units=32):
        self.v4 = v4_obj
        self.n_units = n_units
        self.prototypes = []  # list of vectors (len = Nv4)
        self.prototype_labels = []

    def store_prototypes(self, prototype_vectors, labels=None):
        """
        prototype_vectors: list of 1D arrays (len Nv4) representing the pattern of V4 responses
        """
        self.prototypes = [np.asarray(p, dtype=float) for p in prototype_vectors]
        if labels is None:
            self.prototype_labels = list(range(len(self.prototypes)))
        else:
            self.prototype_labels = labels

    def process(self, video_gray):
        # get V4 activation counts per unit (simple count-based descriptor)
        v4_spikes = self.v4.process(video_gray)
        Nv4 = len(v4_spikes)
        # make a vector: counts per V4 unit
        vec = np.array([len(s) for s in v4_spikes], dtype=float)
        if len(self.prototypes) == 0:
            # no prototypes stored - fallback to simple threshold
            return [np.array([0.0]) if vec.sum() > 0 else np.array([]) for _ in range(self.n_units)]
        # compute similarity (cosine)
        sims = np.array([np.dot(vec, p) / (np.linalg.norm(vec) * np.linalg.norm(p) + 1e-12) for p in self.prototypes])
        # find best match
        best = np.argmax(sims)
        best_score = sims[best]
        # threshold to decide if IT should spike for that prototype
        out = []
        for i in range(self.n_units):
            if best_score > 0.3:
                # fire a brief pattern: time 0 + scaled by similarity
                out.append(np.array([0.0 + 0.001 * best_score]))
            else:
                out.append(np.array([]))
        return out


# ---------- High-level Visual Cortex combining all layers ----------
class VisualCortex:
    def __init__(self, H=64, W=64, fs=100.0):
        self.H = H
        self.W = W
        self.fs = fs
        # low-level components
        self.lgn = LGN(cs_size=15, fs=fs)
        self.v1 = None  # set up later by builder
        # build V1 grid defaults
        neuron_params = {'C': [200e-12, 200e-12], 'g_L': [10e-9, 10e-9], 'E_L': -65e-3, 'g_Na': 1200e-9, 'E_Na': 50e-3,
                         'g_K': 360e-9, 'E_K': -77e-3, 'g_c': 5e-9, 'dt': 1.0 / fs}
        syn_params = {'U': 0.4, 'tau_rec': 0.4, 'tau_fac': 0.0, 'dt': 1.0 / fs}
        self.v1 = self.build_v1(H, W, fs, n_units=36, neuron_params=neuron_params, syn_params=syn_params)
        self.v2 = V2(self.v1, n_units=48)
        self.v4 = V4(self.v2, n_units=32)
        self.it = IT(self.v4, n_units=32)

    def build_v1(self, H, W, fs, n_units, neuron_params, syn_params):
        # instantiate V1 like earlier but keep layer separation: L4 array and an L2/3 builder using same centers
        v1 = type("V1_container", (), {})()  # quick dynamic container
        v1.fs = self.fs
        v1.strf = VisualSTRF(spatial_size=21, orientations=8, scales=[3, 6], temporal_taps=9, fs=self.fs)
        # grid & centers
        grid = int(np.sqrt(n_units))
        coords = []
        for i in range(grid):
            for j in range(grid):
                ci = int((i + 0.5) * H / grid)
                cj = int((j + 0.5) * W / grid)
                coords.append((ci, cj))
        v1.units = []
        for k in range(n_units):
            f_idx = k % v1.strf.F
            unit = {'f_idx': f_idx, 'center': coords[k % len(coords)],
                    'neu': MultiCompartmentNeuron(neuron_params.copy()),
                    'syn': ShortTermSynapse(**syn_params)}
            # add receptors
            unit['neu'].add_receptor(
                Receptor(g_max=1.0e-9, E_rev=0.0, tau_rise=0.0008, tau_decay=0.004, location='soma'))
            unit['neu'].add_receptor(Receptor(g_max=0.6e-9, E_rev=0.0, tau_rise=0.004, tau_decay=0.08, location='dend',
                                              voltage_dependent=True))
            v1.units.append(unit)

        # convenience methods: process -> run L4 units and then L2/3 & L5/6 later
        def process(video_gray):
            # compute feature maps once
            feature_maps = v1.strf.apply(video_gray)  # (T',F,H,W)
            T_out = feature_maps.shape[0]
            # L4 step for each unit
            l4_spikes = []
            for u in v1.units:
                # use L4Unit wrapper to use same neuron/syn API but reusing unit objects
                l4u = L4Unit(f_idx=u['f_idx'], center=u['center'], neuron_params=neuron_params,
                             syn_params=syn_params, pool_radius=1)
                # but to reuse the existing MultiCompNeuron we set l4u.neu to u['neu'] and syn to u['syn']
                l4u.neu = u['neu']
                l4u.syn = u['syn']
                spikes = l4u.step(feature_maps, burst_gain=self.lgn.compute_burst_mask(), modulation=1.0)
                l4_spikes.append(spikes)
            return l4_spikes, feature_maps

        v1.process = process
        return v1

    def run(self, video_gray):
        """
        Run full pipeline on video_gray (T,H,W) - single channel luminance.
        Returns dict of layer outputs.
        """
        # LGN
        lgn_out = self.lgn.apply(video_gray)  # M,P,burst_gain
        # V1 L4: feature maps & spikes
        l4_spikes, feature_maps = self.v1.process(video_gray)
        # V2 corners
        v2_spikes = self.v2.process(video_gray)
        # L23: stitch edges -> we create simple L23 units from V1 grid mapping
        # reuse v1 units centers to build L23 units where each L23 pools a small set of L4 indices
        # For simplicity, build L23 pooling indices by grouping sequential v1 units
        Nv1 = len(self.v1.units)
        n_l23 = Nv1  # one L23 per V1 position (can be fewer)
        l23_units = []
        for k in range(n_l23):
            # pick local neighbors in index space
            pres = [(k + delta) % Nv1 for delta in (-1, 0, 1)]
            l23 = L23Unit(presyn_indices=pres, neuron_params=self.v2._default_neuron_params(),
                          syn_params=self.v2._default_syn_params(), v1_reference=self.v1, facilitation_scale=1.2)
            l23_units.append(l23)
        # compute l23 spikes
        T_bins = video_gray.shape[0] - len(self.v1.strf.temporal) + 1
        l4_spike_lists = l4_spikes
        l23_spikes = [u.run(l4_spike_lists, T_bins) for u in l23_units]
        # L5/6 feedback map
        l56 = L56(self.v1, feedback_strength=0.6)
        feedback_map = l56.compute_feedback(l23_spikes, (self.H, self.W))
        # NOTE: to apply feedback we would re-run L4 with modulation per unit center. For brevity we do not rerun here.
        # V4
        v4_spikes = self.v4.process(video_gray)
        # IT
        it_spikes = self.it.process(video_gray)
        return {'LGN': lgn_out, 'L4': l4_spikes, 'L23': l23_spikes, 'V2': v2_spikes, 'V4': v4_spikes, 'IT': it_spikes}


# ---------- Demo usage ----------
if __name__ == "__main__":
    T, H, W = 60, 64, 64
    frames = np.clip(np.random.randn(T, H, W) * 0.05 + 0.5, 0.0, 1.0)
    pipeline = VisualCortex(H=H, W=W, fs=100.0)
    out = pipeline.run(frames)
    print("V1 L4 spikes counts (first 8 units):", [len(s) for s in out['L4'][:8]])
    print("V2 corner spikes (first 8):", [len(s) for s in out['V2'][:8]])
    print("IT active units (first 8):", [len(s) for s in out['IT'][:8]])
