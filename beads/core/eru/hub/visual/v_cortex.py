import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve, convolve2d

from beads.core.eru.interneuron import ShortTermSynapse, MultiCompartmentNeuron, Receptor


# ---------------------------
# Biologically-grounded Gabor / Visual STRF bank (multi-spectral)
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
    """
    Multi-spectral STRF bank. Applies same spatial filters to each spectral channel (L/M/S),
    then performs temporal filtering (separable assumption). Returns feature maps with
    channel-awareness so downstream units can form single- or double-opponency.
    """

    def __init__(self, spatial_size=21, orientations=8, scales=None, temporal_taps=9,
                 temporal_sigma=0.008, modulation_rate=4.0, fs=100.0, spectral_bands=3):
        if scales is None:
            scales = [3, 6]  # small and medium receptive fields
        self.spatial_size = spatial_size
        self.orientations = orientations
        self.spectral_bands = spectral_bands  # typically 3: L,M,S or derived channels
        bank = []
        for sigma in scales:
            for i in range(orientations):
                theta = i * np.pi / orientations
                lam = max(2.0, sigma)
                bank.append(make_gabor(spatial_size, sigma, theta, lam))
        bank = np.stack([b / (np.linalg.norm(b) + 1e-12) for b in bank], axis=0)  # (F_spat, H, W)
        # We will apply these same spatial kernels to each spectral band (biologically plausible: same orientation filters in chromatic & achromatic streams)
        self.bank = bank
        self.F_spat = bank.shape[0]
        # temporal kernel (same separable assumption across spectral bands)
        times = np.arange(temporal_taps) / fs
        delay = (temporal_taps // 2) / fs
        gauss = np.exp(-0.5 * ((times - delay) / temporal_sigma) ** 2)
        carrier = np.cos(2 * np.pi * modulation_rate * (times - delay))
        tmp = gauss * carrier
        self.temporal = tmp / (np.linalg.norm(tmp) + 1e-12)
        # Final feature count = F_spat * spectral_bands
        self.F = self.F_spat * self.spectral_bands

    def function(self, channels):
        """
        channels: (T, H, W, C) where C == spectral_bands (eg [M, P, K] or [L,M,S]).
        returns: (T_out, F, H, W)
        """
        if channels.ndim != 4:
            raise ValueError("channels must be (T,H,W,C)")
        T, H, W, C = channels.shape
        assert C == self.spectral_bands
        F_spat = self.F_spat
        # compute spatial conv for every (t, f_spat, channel)
        spat_maps = np.zeros((T, C, F_spat, H, W))
        for t in range(T):
            for c in range(C):
                for f in range(F_spat):
                    # boundary='symm' approximates cortical wrap/edge invariance
                    spat_maps[t, c, f] = convolve2d(channels[t, :, :, c], self.bank[f], mode='same', boundary='symm')
        # temporal conv per (channel,f_spat, i,j)
        Tt = len(self.temporal)
        T_out = T - Tt + 1
        out = np.zeros((T_out, self.F, H, W))
        # flatten (channel, f_spat) into feature axis
        feat_idx = 0
        for c in range(C):
            for f in range(F_spat):
                # for each spatial position, convolve across time
                for i in range(H):
                    for j in range(W):
                        out[:, feat_idx, i, j] = np.convolve(spat_maps[:, c, f, i, j], self.temporal, mode='valid')
                feat_idx += 1
        return out


# ---------------------------
# Helper: bin spikes -> matrix
# ---------------------------
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


def dsgc_spike_lists_to_motion_maps(dsgc_spike_lists,
                                    T_bins,
                                    dt,
                                    H,
                                    W,
                                    dir_count,
                                    tile_stride=(1, 1),
                                    smoothing_sigma=1.0,
                                    normalize=True):
    """
    Convert DSGC spike lists into motion-rate maps.
    - dsgc_spike_lists: list of length D (directions). Each element is a list of per-DSGC arrays (spike times in s).
                         Each DSGC index should map spatially (e.g., DSGC i -> pixel/tile i).
    - T_bins, dt: temporal bins and bin width (seconds).
    - H,W: target map spatial dims. If DSGCs are densely per-pixel, H*W == len(per-direction lists). If DSGCs are tiled,
           map accordingly (you can provide mapping externally).
    - dir_count: number of directions in dsgc_spike_lists
    - tile_stride: (tile_h, tile_w) used when transforming DSGC pixel list into HxW; if DSGC list length == (H/tile_h)*(W/tile_w)
    Returns:
      motion_maps: array (T_bins, dir_count, H, W) with smoothed rates (Hz)
    """
    D = dir_count
    Td = T_bins
    motion_maps = np.zeros((Td, D, H, W), dtype=float)

    # assume per-direction lists are length = (H * W) / (tile_h*tile_w) OR equal to H*W
    for d in range(D):
        lists = dsgc_spike_lists[d]
        n_cells = len(lists)
        # arrange into a grid if sizes match, otherwise tile mapping must be given
        if n_cells == H * W:
            idx = 0
            for i in range(H):
                for j in range(W):
                    times = lists[idx]
                    idx += 1
                    if len(times):
                        bins = np.floor(np.asarray(times) / dt).astype(int)
                        bins = bins[(bins >= 0) & (bins < Td)]
                        if bins.size:
                            counts = np.bincount(bins, minlength=Td)
                            motion_maps[:, d, i, j] = counts / dt  # convert to Hz
        else:
            # fallback: evenly distribute DSGC cells across image tiles
            tile_h, tile_w = tile_stride
            cells_per_tile = max(1, n_cells // ((H // tile_h) * (W // tile_w)))
            idx = 0
            for ih in range(0, H, tile_h):
                for jw in range(0, W, tile_w):
                    # pool cells_per_tile DSGCs into the tile
                    tile_counts = np.zeros(Td, dtype=float)
                    for c in range(cells_per_tile):
                        if idx >= n_cells:
                            break
                        times = lists[idx]
                        idx += 1
                        if len(times):
                            bins = np.floor(np.asarray(times) / dt).astype(int)
                            bins = bins[(bins >= 0) & (bins < Td)]
                            if bins.size:
                                tile_counts += np.bincount(bins, minlength=Td)
                    # fill tile
                    i1 = min(H, ih + tile_h)
                    j1 = min(W, jw + tile_w)
                    rate_map = tile_counts / dt  # Hz
                    # optionally divide across tile pixels
                    for i2 in range(ih, i1):
                        for j2 in range(jw, j1):
                            motion_maps[:, d, i2, j2] = rate_map / max(1, cells_per_tile)

    # smoothing & normalization
    for d in range(D):
        for t in range(Td):
            motion_maps[t, d] = gaussian_filter(motion_maps[t, d], sigma=smoothing_sigma)

    if normalize:
        # scale each direction map to unit max across time & space (keeps numerical stability)
        maxv = motion_maps.max(axis=(0, 2, 3), keepdims=True) + 1e-12
        motion_maps = motion_maps / maxv

    return motion_maps  # (T_bins, D, H, W)


# ---------------------------
# LGN with explicit M/P/K streams and tonic/burst gating
# ---------------------------
class LGN:
    """
    LGN: center-surround spatial filtering + separate temporal kernels:
      - M (magno-like): fast transient kernel, sensitive to luminance changes
      - P (parvo-like): sustained, color-opponent (L-M)
      - K (konio-like): S-cone pathway (S - (L+M)) often slower
    Maintains short history to bias M cells into burst mode after prolonged silence.
    """

    def __init__(self, cs_size=15, cs_sigma_c=0.9, cs_sigma_s=3.0, fs=100.0,
                 history_len=40, burst_gain=2.0):
        xs = np.arange(cs_size) - cs_size // 2
        xv, yv = np.meshgrid(xs, xs)
        # center-surround difference of Gaussians (spatial opponency)
        kern = (np.exp(-(xv ** 2 + yv ** 2) / (2 * cs_sigma_c ** 2)) -
                np.exp(-(xv ** 2 + yv ** 2) / (2 * cs_sigma_s ** 2)))
        self.cs_kernel = kern / (np.sum(np.abs(kern)) + 1e-12)
        self.fs = fs
        # biologically-inspired temporal kernels (normalized)
        tM = np.arange(0, 0.05, 1.0 / fs)
        tP = np.arange(0, 0.12, 1.0 / fs)
        # magno: transient biphasic-like kernel approximated with difference of exponentials
        self.kernel_M = (np.exp(-tM / 0.007) * (1 - np.exp(-tM / 0.002)))
        self.kernel_M /= (np.linalg.norm(self.kernel_M) + 1e-12)
        # parvo: slower sustained
        self.kernel_P = np.exp(-tP / 0.03)
        self.kernel_P /= (np.linalg.norm(self.kernel_P) + 1e-12)
        # konio: slower still (S-cone)
        self.kernel_K = np.exp(-tP / 0.05)
        self.kernel_K /= (np.linalg.norm(self.kernel_K) + 1e-12)
        self.history_len = history_len
        self.burst_gain_scalar = burst_gain
        self.history_buffer = []  # circular list of recent CS energies

    def center_surround(self, frame):
        return fftconvolve(frame, self.cs_kernel, mode='same')

    def _update_history(self, cs):
        # cs: (T, H, W)
        energy = np.sqrt(np.mean(cs ** 2, axis=(1, 2)))  # per-frame RMS energy
        if len(self.history_buffer) == 0:
            self.history_buffer = energy[-self.history_len:].tolist() if len(
                energy) >= self.history_len else energy.tolist()
        else:
            self.history_buffer.extend(energy.tolist())
            self.history_buffer = self.history_buffer[-self.history_len:]

    def compute_burst_factor(self):
        # If recent mean activity is low, M cells are more likely to burst (T-type Ca2+ deinactivation)
        if len(self.history_buffer) == 0:
            return 1.0
        recent = np.array(self.history_buffer)
        mean_act = recent.mean()
        thresh = 0.5 * (recent.max() + recent.min() + 1e-12)
        if mean_act < thresh:
            return self.burst_gain_scalar
        return 1.0

    def function(self, frames_rgb_like):
        """
        frames_rgb_like: expected (T, H, W, 3) raw cone-like signals or (L,M,S) / or luminance and chromatic channels.
        Returns dict with 'M','P','K','burst_gain','cs' where each stream is (T_out,H,W).
        """
        # First compute center-surround on luminance-like combination (L+M)
        # If user provides L,M,S, assume channels [:,:,:,0:3] correspond to those cones
        if frames_rgb_like.ndim != 4 or frames_rgb_like.shape[3] < 3:
            raise ValueError("LGN.process expects (T,H,W,3) channels (L,M,S-like).")
        T, H, W, C = frames_rgb_like.shape
        # derive simple cone combinations
        L = frames_rgb_like[:, :, :, 0]
        M = frames_rgb_like[:, :, :, 1]
        S = frames_rgb_like[:, :, :, 2]
        # luminance (magno-dominant): L+M
        lum = 0.5 * (L + M)
        # color-opponent parvo: L - M
        parvo = L - M
        # konio (blue-yellow): S - 0.5*(L+M)
        konio = S - 0.5 * (L + M)
        # compute center-surround on luminance stream (most robust)
        cs = np.stack([self.center_surround(lum[t]) for t in range(T)])  # (T,H,W)
        # update history for burst gating
        self._update_history(cs)
        burst_factor = self.compute_burst_factor()
        # temporal conv for each pixel
        Tm = len(self.kernel_M)
        Tp = len(self.kernel_P)
        Tk = len(self.kernel_K)
        T_out = T - max(Tm, Tp, Tk) + 1
        M_out = np.zeros((T_out, H, W))
        P_out = np.zeros((T_out, H, W))
        K_out = np.zeros((T_out, H, W))
        for i in range(H):
            for j in range(W):
                # magno on center-surround luminance
                M_full = np.convolve(cs[:, i, j], self.kernel_M, mode='valid')
                M_out[:, i, j] = M_full[:T_out]
                # parvo on parvo signal (single-opponent)
                P_full = np.convolve(parvo[:, i, j], self.kernel_P, mode='valid')
                P_out[:, i, j] = P_full[:T_out]
                # konio
                K_full = np.convolve(konio[:, i, j], self.kernel_K, mode='valid')
                K_out[:, i, j] = K_full[:T_out]
        # return streams and scalar burst factor (applied by L4 for gain)
        return {'M': M_out, 'P': P_out, 'K': K_out, 'burst_gain': burst_factor, 'cs': cs}


# ---------------------------
# V1 L4 simple cell (supports luminance, single-opponent, chromatic sampling)
# ---------------------------
class L4Unit:
    """
    L4 unit that can be one of:
      - 'luminance' : driven primarily by M stream features (orientation + temporal)
      - 'single'    : single-opponent color (responds to patches of particular cone sign)
      - 'chromatic' : oriented chromatic (parvo-driven orientation)
    The unit has its own MultiCompartmentNeuron and ST synapse.
    """

    def __init__(self, f_idx, center, neuron_params, syn_params, pool_radius=1, amp=30.0, cell_type='luminance'):
        self.f_idx = f_idx
        self.center = center  # (i,j)
        self.neu = MultiCompartmentNeuron(neuron_params.copy())
        # canonical receptors: fast AMPA at soma, slow NMDA at dendrite (voltage-dependent)
        self.neu.add_receptor(
            Receptor(g_max=1.0e-9, E_rev=0.0, tau_rise=0.0008, tau_decay=0.004, location='soma', name='AMPA'))
        self.neu.add_receptor(
            Receptor(g_max=0.6e-9, E_rev=0.0, tau_rise=0.004, tau_decay=0.08, location='dend', name='NMDA',
                     voltage_dependent=True))
        self.syn = ShortTermSynapse(**syn_params)
        self.pool_radius = pool_radius
        self.amp = amp
        self.cell_type = cell_type  # 'luminance'|'single'|'chromatic'

    def receptive_drive(self, feature_maps):
        # feature_maps: (T, F, H, W) where F encodes channel x spatial-filter
        T, F, H, W = feature_maps.shape
        i, j = self.center
        r = self.pool_radius
        i0, i1 = max(0, i - r), min(H, i + r + 1)
        j0, j1 = max(0, j - r), min(W, j + r + 1)
        patch = feature_maps[:, :, i0:i1, j0:j1]  # (T, F, h, w)
        # choose which feature indices to emphasize depending on cell type:
        # Assume feature packing: [ (spat_f x channel0), (spat_f x channel1), ... ]
        # We map f_idx to the spatial filter index (0..F_spat-1) and choose channel slices accordingly.
        # To keep it simple: compute mean over all features but weight by channel preference
        # channel_count inferred:
        # If F = F_spat * C => C = F // F_spat
        # f_idx provided as spatial filter index (0..F_spat-1)
        # We'll compute channel slices
        # NOTE: in build_v1 we assign f_idx appropriately
        F_spat = None
        # compute F_spat from feature_maps shape and expected channels
        # We assume the VisualSTRF packing convention: channel blocks of size F_spat
        # i.e., feature index = channel*F_spat + spat_idx
        # infer channels:
        # (this is somewhat defensive; you can pass explicit metadata if you want)
        # here we compute channel-weighted drive: prefer M for luminance, P for chromatic single-opponent, K for konio
        _, totalF, _, _ = feature_maps.shape
        # Heuristic: assume F_spat divides totalF into equal channel blocks (common case)
        # If not divisible, fallback to mean across features
        # We'll compute mean per feature column and then reweight
        feat_mean = patch.reshape(T, totalF, -1).mean(axis=2)  # (T, totalF)
        drive = feat_mean.mean(axis=1)
        return np.maximum(drive, 0.0)

    def function(self, feature_maps, burst_gain=1.0, modulation=1.0):
        drive = self.receptive_drive(feature_maps)  # (T,)
        # firing rate scaling -> probability per timestep
        lam = (drive / (np.max(drive) + 1e-12)) * self.amp * modulation * burst_gain
        p = 1.0 - np.exp(-lam * self.neu.dt)
        prespikes = (np.random.rand(p.size) < p).astype(float)
        rel = self.syn.step(prespikes)
        syn_rel = np.stack([rel * 1.0, rel * 0.6], axis=1)  # AMPA weight 1.0, NMDA weight 0.6
        times, Vs = self.neu.step(syn_rel)
        thr = -30e-3
        idx = np.where(Vs >= thr)[0]
        return np.unique(idx) * self.neu.dt  # spike times (s)


# ---------------------------
# L2/3 double-opponency & horizontal facilitation
# ---------------------------
class L23Unit:
    """
    Implements a double-opponent pooling cell that receives multiple L4 presynaptic spike lists.
    Uses ShortTermSynapse for each presynaptic input and transiently increases facilitation ('u')
    when neighbors are coactive (horizontal facilitation).
    """

    def __init__(self, presyn_indices, neuron_params, syn_params, v1_reference, facilitation_scale=1.2):
        self.presyn = np.array(presyn_indices, dtype=int)
        self.syns = [ShortTermSynapse(**syn_params) for _ in self.presyn]
        self.neu = MultiCompartmentNeuron(neuron_params.copy())
        # canonical receptors (AMPA + NMDA)
        self.neu.add_receptor(
            Receptor(g_max=1.0e-9, E_rev=0.0, tau_rise=0.0008, tau_decay=0.004, location='soma', name='AMPA'))
        self.neu.add_receptor(
            Receptor(g_max=0.7e-9, E_rev=0.0, tau_rise=0.004, tau_decay=0.08, location='dend', name='NMDA',
                     voltage_dependent=True))
        self.v1 = v1_reference
        self.facilitation_scale = facilitation_scale

    def function(self, l4_spike_lists, T_bins):
        dt = 1.0 / self.v1.fs
        l4_mat = bin_spikes_to_matrix(l4_spike_lists, T_bins, dt)  # (N_l4, T_bins)
        drive = np.zeros(T_bins, dtype=float)
        # horizontal facilitation: when multiple presyn indices fire together, transiently boost local release
        local_sum = np.sum(l4_mat[self.presyn], axis=0)
        fac = 1.0 + (self.facilitation_scale - 1.0) * (local_sum > 1.0).astype(float)
        for k, idx in enumerate(self.presyn):
            s = l4_mat[idx]
            rel = self.syns[k].step(s * fac)
            drive += rel
        # To create double-opponency we here assume some presyn inputs carry opposite signs;
        # sign mixing would be implemented by assigning negative weights to synapses in used presyn list.
        syn_rel = np.stack([drive, 0.6 * drive], axis=1)
        times, Vs = self.neu.step(syn_rel)
        thr = -30e-3
        idx = np.where(Vs >= thr)[0]
        return np.unique(idx) * self.neu.dt


# ---------------------------
# L5/6 feedback (corticothalamic gain map)
# ---------------------------
class L56Unit:
    def __init__(self, v1_reference, feedback_strength=0.5):
        self.v1 = v1_reference
        self.feedback_strength = feedback_strength

    def function(self, l23_spikes, video_shape):
        rates = np.array([len(s) for s in l23_spikes], dtype=float)
        if rates.size == 0 or rates.max() == 0:
            return np.ones(video_shape, dtype=float)
        norm = rates / (np.max(rates) + 1e-12)
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


# ---------------------------
# V2 corner detectors (coincidence-based)
# ---------------------------
class V2UnitCorner:
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

    def function(self, l4_spikes, T_bins):
        dt = 1.0 / self.v1.fs
        a = bin_spikes_to_matrix([l4_spikes[self.idx_a]], T_bins, dt)[0]
        b = bin_spikes_to_matrix([l4_spikes[self.idx_b]], T_bins, dt)[0]
        rel_a = self.syn_a.step(a)
        rel_b = self.syn_b.step(b)
        drive = rel_a * rel_b  # multiplicative/coincidence detection
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
        for _ in range(n_units):
            a = int(rng.integers(0, Nv1))
            b = (a + int(rng.integers(1, max(2, Nv1 // 8)))) % Nv1
            unit = V2UnitCorner(a, b, neuron_params=self._default_neuron_params(),
                                syn_params=self._default_syn_params(), v1_ref=self.v1)
            self.units.append(unit)

    def _default_neuron_params(self):
        return {'C': [200e-12, 200e-12], 'g_L': [10e-9, 10e-9], 'E_L': -65e-3,
                'g_Na': 1200e-9, 'E_Na': 50e-3, 'g_K': 360e-9, 'E_K': -77e-3,
                'g_c': 5e-9, 'dt': 1.0 / self.v1.fs}

    def _default_syn_params(self):
        return {'U': 0.4, 'tau_rec': 0.4, 'tau_fac': 0.0, 'dt': 1.0 / self.v1.fs}

    def function(self, channels):
        # channels here not needed directly; we use l4 outputs below in VisualCortex
        raise NotImplementedError("V2.process used via VisualCortex orchestration.")


# ---------------------------
# V4: simple pooling and light-invariance normalization
# ---------------------------
class V4:
    def __init__(self, v2_obj, n_units=32):
        self.v2 = v2_obj
        self.n_units = n_units
        Nv2 = max(1, len(self.v2.units))
        rng = np.random.default_rng(11)
        self.units = []
        for _ in range(n_units):
            K = min(12, Nv2)
            pres = rng.choice(range(Nv2), size=K, replace=False).tolist()
            w = rng.normal(1.0, 0.25, size=K)
            u = {'pres': pres, 'w': w,
                 'neu': MultiCompartmentNeuron(self.v2._default_neuron_params())}
            u['neu'].add_receptor(Receptor(g_max=1.2e-9, E_rev=0.0, tau_rise=0.0008, tau_decay=0.004, location='soma'))
            u['neu'].add_receptor(Receptor(g_max=0.8e-9, E_rev=0.0, tau_rise=0.004, tau_decay=0.08, location='dend',
                                           voltage_dependent=True))
            u['syns'] = [ShortTermSynapse(**self.v2._default_syn_params()) for _ in pres]
            self.units.append(u)

    def function(self, v2_spikes, video_gray):
        # v2_spikes: list of spike arrays from V2 units
        T_bins = video_gray.shape[0] - len(self.v2.v1.strf.temporal) + 1
        dt = 1.0 / self.v2.v1.fs
        v2_mat = bin_spikes_to_matrix(v2_spikes, T_bins, dt)
        out = []
        for u in self.units:
            drive = np.zeros(T_bins, dtype=float)
            for k, idx in enumerate(u['pres']):
                rel = u['syns'][k].step(v2_mat[idx])
                drive += u['w'][k] * rel
            denom = np.mean(drive) + 1e-12
            drive_norm = drive / denom
            syn_rel = np.stack([drive_norm, 0.6 * drive_norm], axis=1)
            times, Vs = u['neu'].step(syn_rel)
            thr = -30e-3
            idx_sp = np.where(Vs >= thr)[0]
            out.append(np.unique(idx_sp) * u['neu'].dt)
        return out


# ---------------------------
# IT: prototype associative readout
# ---------------------------
class IT:
    def __init__(self, v4_obj, n_units=32):
        self.v4 = v4_obj
        self.n_units = n_units
        self.prototypes = []
        self.prototype_labels = []

    def store_prototypes(self, prototype_vectors, labels=None):
        self.prototypes = [np.asarray(p, dtype=float) for p in prototype_vectors]
        if labels is None:
            self.prototype_labels = list(range(len(self.prototypes)))
        else:
            self.prototype_labels = labels

    def function(self, v4_spikes):
        vec = np.array([len(s) for s in v4_spikes], dtype=float)
        if len(self.prototypes) == 0:
            return [np.array([0.0]) if vec.sum() > 0 else np.array([]) for _ in range(self.n_units)]
        sims = np.array([np.dot(vec, p) / (np.linalg.norm(vec) * np.linalg.norm(p) + 1e-12) for p in self.prototypes])
        best = np.argmax(sims)
        best_score = sims[best]
        out = []
        for i in range(self.n_units):
            if best_score > 0.3:
                out.append(np.array([0.0 + 0.001 * best_score]))
            else:
                out.append(np.array([]))
        return out


# ---------------------------
# High-level Visual Cortex orchestrator
# ---------------------------
class VisualCortex:
    """
    Orchestrates LGN -> V1(L4,L23,L56) -> V2 -> V4 -> IT.
    Channels convention: input frames should be (T,H,W,3) representing L,M,S (or L-like, M-like, S-like).
    """

    def __init__(self, H=64, W=64, fs=100.0, n_v1_units=36):
        self.H = H
        self.W = W
        self.fs = fs
        self.lgn = LGN(cs_size=15, fs=fs)
        # neuron + syn params (default biophysical values)
        neuron_params = {'C': [200e-12, 200e-12], 'g_L': [10e-9, 10e-9], 'E_L': -65e-3,
                         'g_Na': 1200e-9, 'E_Na': 50e-3, 'g_K': 360e-9, 'E_K': -77e-3, 'g_c': 5e-9, 'dt': 1.0 / fs}
        syn_params = {'U': 0.4, 'tau_rec': 0.4, 'tau_fac': 0.0, 'dt': 1.0 / fs}
        # build V1
        self.v1 = self.build_v1(None, H, W, fs, n_units=n_v1_units, neuron_params=neuron_params, syn_params=syn_params)
        # V2, V4, IT
        self.v2 = V2(self.v1, n_units=48)
        self.v4 = V4(self.v2, n_units=32)
        self.it = IT(self.v4, n_units=32)

    def build_v1(self, dsgc_spike_lists, H, W, fs, n_units, neuron_params, syn_params):
        v1 = type("V1_container", (), {})()
        v1.fs = self.fs
        # VisualSTRF: F_spat * C features (C=3: M,P,K streams)
        v1.strf = VisualSTRF(spatial_size=21, orientations=8, scales=[3, 6], temporal_taps=9, fs=self.fs,
                             spectral_bands=3)
        grid = int(np.sqrt(n_units))
        coords = []
        for i in range(grid):
            for j in range(grid):
                ci = int((i + 0.5) * H / grid)
                cj = int((j + 0.5) * W / grid)
                coords.append((ci, cj))
        v1.units = []
        # create units tiled over the grid; alternate cell types for diversity
        for k in range(n_units):
            f_idx = k % v1.strf.F_spat  # spatial-filter index; actual feature index = channel*F_spat + f_idx
            center = coords[k % len(coords)]
            cell_type = 'luminance' if (k % 3) == 0 else ('single' if (k % 3) == 1 else 'chromatic')
            unit = {'f_idx': f_idx, 'center': center,
                    'neu': MultiCompartmentNeuron(neuron_params.copy()),
                    'syn': ShortTermSynapse(**syn_params),
                    'cell_type': cell_type}
            # receptors (matching L4Unit)
            unit['neu'].add_receptor(
                Receptor(g_max=1.0e-9, E_rev=0.0, tau_rise=0.0008, tau_decay=0.004, location='soma'))
            unit['neu'].add_receptor(Receptor(g_max=0.6e-9, E_rev=0.0, tau_rise=0.004, tau_decay=0.08, location='dend',
                                              voltage_dependent=True))
            v1.units.append(unit)

        def function(frames_lms, dsgc_spike_lists=None, dsgc_dir_count=None, dsgc_tile_stride=(1,1),
                     dsgc_smoothing_sigma=1.0, dsgc_normalize=True):
            """
            frames_lms: (T,H,W,3) input (L,M,S)
            dsgc_spike_lists: optional list of length D (directions). Each element is a list of per-DSGC arrays (spike times in s).
                               If None, no motion channel is added.
            dsgc_dir_count: optional int (D). If None, inferred from dsgc_spike_lists.
            returns: l4_spike_lists, feature_maps (feature_maps will include appended motion features if DSGCs supplied)
            """
            # 1) LGN -> get M,P,K streams (T_out,H,W)
            lgn_out = self.lgn.function(frames_lms)
            T_out = lgn_out['M'].shape[0]     # temporal length after LGN temporal filtering
            # pack streams into channels (T_out,H,W,3) ordering [M,P,K]
            channels = np.stack([lgn_out['M'], lgn_out['P'], lgn_out['K']], axis=-1)  # (T_out,H,W,3)

            # 2) If no DSGC motion input provided, compute feature maps and exit normally
            #    Otherwise compute motion_maps aligned to LGN frame grid, convert to motion features,
            #    and append those features to feature_maps (so VisualSTRF doesn't need to be re-instantiated).
            # compute base feature maps from color streams
            feature_maps = v1.strf.function(channels)   # (T_feat, F_total, H, W)
            T_feat = feature_maps.shape[0]
            F_total = feature_maps.shape[1]
            F_spat = v1.strf.F_spat
            temporal_kernel = v1.strf.temporal
            dt = 1.0 / self.fs

            if dsgc_spike_lists is None:
                # no motion augmentation requested
                l4_spikes = []
                for u in v1.units:
                    l4u = L4Unit(f_idx=u['f_idx'], center=u['center'],
                                 neuron_params=neuron_params, syn_params=syn_params,
                                 pool_radius=1, amp=30.0, cell_type=u['cell_type'])
                    l4u.neu = u['neu']
                    l4u.syn = u['syn']
                    spikes = l4u.function(feature_maps, burst_gain=lgn_out['burst_gain'], modulation=1.0)
                    l4_spikes.append(spikes)
                return l4_spikes, feature_maps

            # --- DSGC provided: build motion maps ---
            # infer direction count
            if dsgc_dir_count is None:
                dsgc_dir_count = len(dsgc_spike_lists)

            # helper: bin DSGC spike lists per direction into (T_out, H, W) rate maps
            # Expect each element of dsgc_spike_lists to be a list of per-DSGC arrays.
            # We'll attempt to tile DSGCs onto the HxW grid if counts match, otherwise use tile_stride pooling.
            from scipy.ndimage import gaussian_filter  # local import safe inside function

            D = dsgc_dir_count
            Td = T_out
            motion_maps = np.zeros((Td, D, self.H, self.W), dtype=float)

            for d in range(D):
                lists = dsgc_spike_lists[d]
                n_cells = len(lists)
                # if a one-to-one mapping exists: n_cells == H*W
                if n_cells == self.H * self.W:
                    idx = 0
                    for i in range(self.H):
                        for j in range(self.W):
                            times = lists[idx] if idx < n_cells else np.array([])
                            idx += 1
                            if len(times):
                                bins = np.floor(np.asarray(times) / dt).astype(int)
                                bins = bins[(bins >= 0) & (bins < Td)]
                                if bins.size:
                                    counts = np.bincount(bins, minlength=Td)
                                    motion_maps[:, d, i, j] = counts / dt
                else:
                    # fallback: tile the image and pool DSGCs per tile
                    tile_h, tile_w = dsgc_tile_stride
                    n_tiles_h = (self.H + tile_h - 1) // tile_h
                    n_tiles_w = (self.W + tile_w - 1) // tile_w
                    cells_per_tile = max(1, n_cells // (n_tiles_h * n_tiles_w))
                    idx = 0
                    for ih in range(0, self.H, tile_h):
                        for jw in range(0, self.W, tile_w):
                            tile_counts = np.zeros(Td, dtype=float)
                            for c in range(cells_per_tile):
                                if idx >= n_cells:
                                    break
                                times = lists[idx]
                                idx += 1
                                if len(times):
                                    bins = np.floor(np.asarray(times) / dt).astype(int)
                                    bins = bins[(bins >= 0) & (bins < Td)]
                                    if bins.size:
                                        tile_counts += np.bincount(bins, minlength=Td)
                            i1 = min(self.H, ih + tile_h)
                            j1 = min(self.W, jw + tile_w)
                            if cells_per_tile > 0:
                                rate_map = tile_counts / (dt * cells_per_tile)
                            else:
                                rate_map = tile_counts / max(1e-12, dt)
                            for i2 in range(ih, i1):
                                for j2 in range(jw, j1):
                                    motion_maps[:, d, i2, j2] = rate_map

            # smoothing & normalize
            if dsgc_smoothing_sigma is not None and dsgc_smoothing_sigma > 0:
                for t in range(Td):
                    for d in range(D):
                        motion_maps[t, d] = gaussian_filter(motion_maps[t, d], sigma=dsgc_smoothing_sigma)
            if dsgc_normalize:
                # normalize each direction's map to unit max
                maxv = motion_maps.max(axis=(0,2,3), keepdims=True) + 1e-12
                motion_maps = motion_maps / maxv

            # collapse direction axis to a single motion channel (simple average across directions)
            motion_channel = motion_maps.mean(axis=1)  # (T_out, H, W)

            # --- convert motion channel into motion feature-block (apply same spatial bank + temporal kernel) ---
            # spatial conv per spatial-filter
            Tt = len(temporal_kernel)
            T_feat_expected = T_out - Tt + 1
            if T_feat_expected != T_feat:
                # consistent alignment: we computed feature_maps from channels (T_out->T_feat); T_feat_expected should match
                T_feat = min(T_feat, T_feat_expected)

            # compute spatial responses of motion_channel for each spatial filter
            # spat_motion: (T_out, F_spat, H, W)
            spat_motion = np.zeros((T_out, F_spat, self.H, self.W))
            for t in range(T_out):
                for f in range(F_spat):
                    spat_motion[t, f] = convolve2d(motion_channel[t], v1.strf.bank[f], mode='same', boundary='symm')

            # temporal conv per (f, i, j) -> motion_features: (T_feat, F_spat, H, W)
            motion_features = np.zeros((T_feat, F_spat, self.H, self.W))
            for f in range(F_spat):
                for i in range(self.H):
                    for j in range(self.W):
                        motion_features[:, f, i, j] = np.convolve(spat_motion[:, f, i, j], temporal_kernel, mode='valid')[:T_feat]

            # reshape motion_features into a feature-block matching packing convention: append along feature axis
            # feature_maps currently (T_feat, F_total, H, W). Append motion block of size F_spat along axis=1
            feature_maps_aug = np.concatenate([feature_maps, motion_features], axis=1)  # new F = F_total + F_spat

            # 3) run each L4 unit on augmented feature maps
            l4_spikes = []
            for u in v1.units:
                l4u = L4Unit(f_idx=u['f_idx'], center=u['center'],
                             neuron_params=neuron_params, syn_params=syn_params,
                             pool_radius=1, amp=30.0, cell_type=u['cell_type'])
                l4u.neu = u['neu']
                l4u.syn = u['syn']
                spikes = l4u.function(feature_maps_aug, burst_gain=lgn_out['burst_gain'], modulation=1.0)
                l4_spikes.append(spikes)

            return l4_spikes, feature_maps_aug


    def function(self, frames_lms):
        """
        frames_lms: (T,H,W,3) input - L,M,S or cone-like channels.
        Returns dictionary with layer outputs (spike lists).
        """
        # V1 L4
        l4_spikes, feature_maps = self.v1.process(frames_lms)
        # L2/3 units: one per V1 position for simplicity
        Nv1 = len(self.v1.units)
        n_l23 = Nv1
        l23_units = []
        for k in range(n_l23):
            pres = [(k + delta) % Nv1 for delta in (-1, 0, 1)]
            l23 = L23Unit(presyn_indices=pres, neuron_params=self.v2._default_neuron_params(),
                          syn_params=self.v2._default_syn_params(), v1_reference=self.v1, facilitation_scale=1.2)
            l23_units.append(l23)
        T_bins = frames_lms.shape[0] - len(self.v1.strf.temporal) + 1
        l23_spikes = [u.process(l4_spikes, T_bins) for u in l23_units]
        # feedback map (not re-applied here for brevity)
        l56 = L56Unit(self.v1, feedback_strength=0.6)
        feedback_map = l56.function(l23_spikes, (self.H, self.W))
        # V2 corners (callers rely on VisualCortex orchestration)
        # For V2, compute spikes by mapping l4 spikes -> v2 units
        v2_spikes = []
        for u in self.v2.units:
            # each V2UnitCorner needs l4 spikes and T_bins
            # reuse existing objects
            v2_spikes.append(u.process(l4_spikes, T_bins))
        # V4 pooling
        v4_spikes = self.v4.function(v2_spikes, frames_lms[..., 0])  # pass a representative luminance for normalization
        # IT readout
        it_spikes = self.it.function(v4_spikes)
        return {'LGN': None, 'L4': l4_spikes, 'L23': l23_spikes, 'V2': v2_spikes, 'V4': v4_spikes, 'IT': it_spikes}


# TODO: Temporary code block to test these cells. Input and output should be through files (which can be used for the demo)
# ---------------------------
# Minimal demo (commented)
# ---------------------------
"""
if __name__ == '__main__':
    T, H, W = 120, 64, 64
    # create synthetic L,M,S-like channels (small random variations)
    frames = np.clip(np.random.randn(T, H, W, 3) * 0.02 + 0.5, 0.0, 1.0)
    vc = VisualCortex(H=H, W=W, fs=100.0, n_v1_units=36)
    out = vc.process(frames)
    print('L4 spike counts (first 8):', [len(s) for s in out['L4'][:8]])
    print('V2 spike counts (first 8):', [len(s) for s in out['V2'][:8]])
    print('IT activations (first 8):', [len(s) for s in out['IT'][:8]])
"""
