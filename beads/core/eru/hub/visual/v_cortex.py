import argparse
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve, convolve2d

from beads.core.eru.interneuron import ShortTermSynapse, MultiCompartmentNeuron, Receptor


# TODO: Revamp this code to make it simpler
"""
You have a 5 dimensional vector (flattened 2D retina space, time, M/P/K/DSGC channels and a bank of temporal 
phases) which needs to be used to inform (direction, collinearity, etc) various parts of the visual 
cortex. 

A multicompartment LIF neuron is a good bridge spatially and across channels if the tuning is 
done properly. However, since we don't incorporate genetics or neurotransmitters yet, post demo finetuning
is probably the only option left. A structural segment-by-segment output vs expectation analysis would
alleviate the finetuning remaining in the system.
"""

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

def rgc_spikes_to_frames(rgc_dict, T_bins, dt, H, W, tile_stride=(1,1)):
    """
    Convert RGC spike lists into pseudo-frames (T_bins, H, W, 3) ordered as [M, P, K].
    rgc_dict: dict with optional keys 'parasol', 'midget', 'small_bi'. Each value is a list of per-cell spike-arrays.
    If a channel's number of cells == H*W we map 1:1; otherwise we tile the cells into image tiles of size tile_stride.
    Returns frames_lms shape (T_bins, H, W, 3)
    """
    Td = T_bins
    frames = np.zeros((Td, H, W, 3), dtype=float)  # M,P,K

    def fill_channel(lists, channel_idx):
        if lists is None or len(lists) == 0:
            return
        n_cells = len(lists)
        # quick bin -> matrix (n_cells, T)
        mat = bin_spikes_to_matrix(lists, Td, dt)  # rows=cells, cols=time
        if n_cells == H * W:
            # one-to-one mapping
            idx = 0
            for i in range(H):
                for j in range(W):
                    frames[:, i, j, channel_idx] = mat[idx] / dt  # Hz
                    idx += 1
        else:
            # tile pooling over tile_stride
            tile_h, tile_w = tile_stride
            n_tiles_h = (H + tile_h - 1) // tile_h
            n_tiles_w = (W + tile_w - 1) // tile_w
            cells_per_tile = max(1, n_cells // (n_tiles_h * n_tiles_w))
            idx = 0
            for ih in range(0, H, tile_h):
                for jw in range(0, W, tile_w):
                    tile_counts = np.zeros(Td, dtype=float)
                    for c in range(cells_per_tile):
                        if idx >= n_cells:
                            break
                        tile_counts += mat[idx]
                        idx += 1
                    i1 = min(H, ih + tile_h)
                    j1 = min(W, jw + tile_w)
                    if cells_per_tile > 0:
                        rate_map = tile_counts / (dt * max(1, cells_per_tile))
                    else:
                        rate_map = tile_counts / max(1e-12, dt)
                    for i2 in range(ih, i1):
                        for j2 in range(jw, j1):
                            frames[:, i2, j2, channel_idx] = rate_map

    # channels mapping
    fill_channel(rgc_dict.get('parasol'), 0)   # M (magno-like)
    fill_channel(rgc_dict.get('midget'), 1)    # P (parvo-like)
    fill_channel(rgc_dict.get('small_bi'), 2)  # K (konio-like)

    # Optionally normalize per-channel if desired (keep as rates by default)
    return frames


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
        self.function = None
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

        def function(frames_lms=None, rgc_spike_dict=None, dsgc_spike_lists=None, dsgc_dir_count=None,
                    dsgc_tile_stride=(1,1), dsgc_smoothing_sigma=1.0, dsgc_normalize=True):
            """
            Flexible processing entry:
              - Either provide frames_lms (T,H,W,3) directly, OR
              - provide rgc_spike_dict { 'parasol': [...], 'midget': [...], 'small_bi': [...] }
                along with Td/dt derived below to synthesize frames.
            Optionally provide DSGC spike lists to augment motion features (same semantics as before).
            Returns: l4_spike_lists, feature_maps (motion-augmented if DSGC provided)
            """
            # If spike-based input provided, synthesize frames first
            # We need Td (T_out) and dt to align with LGN temporal kernels:
            # We'll approximate using self.lgn kernels lengths to compute Td.
            # Use a conservative dt = 1.0 / self.fs
            dt = 1.0 / self.fs

            if frames_lms is None:
                if rgc_spike_dict is None:
                    raise ValueError("Either frames_lms or rgc_spike_dict must be provided.")
                # Determine T_bins based on desired output length. Choose T_bins = some length (user should pick)
                # Here we infer Td from the longest kernel length in LGN to be safe:
                max_klen = max(len(self.lgn.kernel_M), len(self.lgn.kernel_P), len(self.lgn.kernel_K))
                # user facing T_bins should be provided in rgc_spike_dict under key '_T_bins' optional; else require an estimate
                T_bins = rgc_spike_dict.get('_T_bins') if isinstance(rgc_spike_dict, dict) and ('_T_bins' in rgc_spike_dict) else None
                if T_bins is None:
                    # fallback infer from latest spike time seen (coarse)
                    max_time = 0.0
                    for ch in ('parasol','midget','small_bi'):
                        lists = rgc_spike_dict.get(ch) if rgc_spike_dict else None
                        if not lists:
                            continue
                        for arr in lists:
                            if len(arr):
                                max_time = max(max_time, np.max(arr))
                    if max_time <= 0.0:
                        # default small duration
                        max_time = 1.0
                    # we produce T_bins such that T_bins*dt >= max_time + max_klen*dt
                    T_bins = int(np.ceil((max_time + max_klen * dt) / dt))
                # build frames (T_bins, H, W, 3)
                frames_lms = rgc_spikes_to_frames(rgc_spike_dict, T_bins, dt, self.H, self.W, tile_stride=dsgc_tile_stride)

            # 1) LGN -> get M,P,K streams (T_out,H,W)
            lgn_out = self.lgn.function(frames_lms)
            T_out = lgn_out['M'].shape[0]     # temporal length after LGN temporal filtering
            # pack streams into channels (T_out,H,W,3) ordering [M,P,K]
            channels = np.stack([lgn_out['M'], lgn_out['P'], lgn_out['K']], axis=-1)  # (T_out,H,W,3)

            # 2) compute base feature maps from color streams
            feature_maps = v1.strf.function(channels)   # (T_feat, F_total, H, W)
            T_feat = feature_maps.shape[0]
            F_total = feature_maps.shape[1]
            F_spat = v1.strf.F_spat
            temporal_kernel = v1.strf.temporal

            # 3) If DSGC (motion) is provided, compute motion features & append (same code as before)
            if dsgc_spike_lists is not None:
                if dsgc_dir_count is None:
                    dsgc_dir_count = len(dsgc_spike_lists)
                D = dsgc_dir_count
                Td = T_out
                motion_maps = np.zeros((Td, D, self.H, self.W), dtype=float)
                # reuse dsgc_spike_lists_to_motion_maps helper to make motion_maps (it handles tiling)
                motion_maps = dsgc_spike_lists_to_motion_maps(dsgc_spike_lists, T_bins=Td, dt=dt, H=self.H, W=self.W,
                                                              dir_count=D, tile_stride=dsgc_tile_stride,
                                                              smoothing_sigma=dsgc_smoothing_sigma, normalize=dsgc_normalize)
                # collapse directions -> motion channel
                motion_channel = motion_maps.mean(axis=1)  # (T_out, H, W)

                # compute spatial responses of motion_channel for each spatial filter
                Tt = len(temporal_kernel)
                T_feat_expected = T_out - Tt + 1
                T_feat = min(T_feat, T_feat_expected)
                spat_motion = np.zeros((T_out, F_spat, self.H, self.W))
                for t in range(T_out):
                    for f in range(F_spat):
                        spat_motion[t, f] = convolve2d(motion_channel[t], v1.strf.bank[f], mode='same', boundary='symm')

                motion_features = np.zeros((T_feat, F_spat, self.H, self.W))
                for f in range(F_spat):
                    for i in range(self.H):
                        for j in range(self.W):
                            motion_features[:, f, i, j] = np.convolve(spat_motion[:, f, i, j], temporal_kernel, mode='valid')[:T_feat]

                # append along feature axis
                feature_maps = np.concatenate([feature_maps[:T_feat], motion_features], axis=1)

            # 4) run each L4 unit on (possibly augmented) feature_maps
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
        self.function = function


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
    out = vc.function(frames)
    print('L4 spike counts (first 8):', [len(s) for s in out['L4'][:8]])
    print('V2 spike counts (first 8):', [len(s) for s in out['V2'][:8]])
    print('IT activations (first 8):', [len(s) for s in out['IT'][:8]])
"""
#!/usr/bin/env python3
"""
demo_vc_from_spikes.py

Minimal demo that:
 - reads simple CSV exports for Parasol, Midget, Small-Bi and DSGC cell lists
 - converts those table rows into synthetic spike-time lists (Poisson / single-spike stylized)
 - runs the VisualCortex pipeline (expects VisualCortex with v1.process(...) available)
 - writes simple output files (L4/V2/IT spike counts and optional NPZ of spike lists)

Usage:
    python demo_vc_from_spikes.py \
        --parasol parasol.csv \
        --midget midget.csv \
        --smallbi small_bi.csv \
        --dsgc dsgc.csv \
        --duration 1.0 \
        --fs 100.0 \
        --outdir /tmp/vc_demo

Notes:
 - Mapping from table fields -> firing rates is heuristic (scale_factor) and configurable.
 - This script assumes your VisualCortex (and helper rgc_spikes_to_frames) are importable
   as `from visual_strf_v1 import VisualCortex` or adjust the import below to point to
   your module that defines VisualCortex.
"""

def poisson_spike_times(rate_hz, duration_s, rng):
    """Return sorted spike times drawn from homogeneous Poisson process."""
    if rate_hz <= 0:
        return np.array([], dtype=float)
    expected_n = rate_hz * duration_s
    n = rng.poisson(expected_n)
    if n <= 0:
        return np.array([], dtype=float)
    times = rng.random(n) * duration_s
    return np.sort(times)


def single_spike_if_flag(flag, duration_s, rng):
    """If flag true-ish, return one random spike time in the window, else empty."""
    if not flag:
        return np.array([], dtype=float)
    return np.array([float(rng.random() * duration_s)])


def rows_to_spike_lists(df, duration_s, fs, scale_factor=10.0, time_field=None, integrated_field=None, flag_field=None):
    """
    Convert DataFrame rows to list of spike arrays.
    - scale_factor: multiplier to map `integrated_field` (or response) to Hz
    - fields are heuristics; if flag_field present use single_spike_if_flag
    """
    rng = np.random.default_rng(0)
    spike_lists = []
    for _, row in df.iterrows():
        # priority: explicit time-field (list/CSV of times) -> integrated_field -> flag_field
        if time_field and time_field in row and pd.notna(row[time_field]):
            # attempt to parse a semicolon/comma-separated list of times
            val = str(row[time_field])
            parts = [p.strip() for p in val.replace(';', ',').split(',') if p.strip() != ""]
            try:
                times = np.array([float(p) for p in parts], dtype=float)
                times = times[(times >= 0) & (times <= duration_s)]
                times = np.sort(times)
            except Exception:
                times = np.array([], dtype=float)
            spike_lists.append(times)
            continue

        if integrated_field and integrated_field in row and pd.notna(row[integrated_field]):
            val = float(row[integrated_field])
            rate = max(0.0, val) * float(scale_factor)  # heuristic mapping -> Hz
            spike_lists.append(poisson_spike_times(rate, duration_s, rng))
            continue

        if flag_field and flag_field in row and pd.notna(row[flag_field]):
            val = row[flag_field]
            # treat any nonzero value as a spike flag
            spike_lists.append(single_spike_if_flag(bool(val), duration_s, rng))
            continue

        # fallback: no useful columns -> empty spike train
        spike_lists.append(np.array([], dtype=float))
    return spike_lists


def build_rgc_dict_from_csv(parasol_csv, midget_csv, smallbi_csv, dsgc_csv,
                            duration_s, fs, scale_parasol=10.0, scale_midget=5.0, scale_smallbi=5.0):
    """
    Read CSVs and build rgc_spike_dict (parasol->M, midget->P, small_bi->K) and dsgc_spike_lists.
    CSVs are expected to have one row per cell; heuristics applied depending on column names:
      - look for 'response' or 'integrated_signal' as continuous drive to map to rate
      - for DSGC look for 'spike_output' or 'integrated_signal'
    """
    parasol_df = pd.read_csv(parasol_csv) if parasol_csv and os.path.exists(parasol_csv) else pd.DataFrame()
    midget_df = pd.read_csv(midget_csv) if midget_csv and os.path.exists(midget_csv) else pd.DataFrame()
    smallbi_df = pd.read_csv(smallbi_csv) if smallbi_csv and os.path.exists(smallbi_csv) else pd.DataFrame()
    dsgc_df = pd.read_csv(dsgc_csv) if dsgc_csv and os.path.exists(dsgc_csv) else pd.DataFrame()

    # Heuristic field names
    cont_fields = ['response', 'integrated_signal', 'response_value', 'rate']
    flag_fields = ['spike_output', 'spike', 'spike_flag']

    def pick_field(df, fields):
        for f in fields:
            if f in df.columns:
                return f
        return None

    parasol_field = pick_field(parasol_df, cont_fields) or pick_field(parasol_df, flag_fields)
    midget_field = pick_field(midget_df, cont_fields) or pick_field(midget_df, flag_fields)
    smallbi_field = pick_field(smallbi_df, cont_fields) or pick_field(smallbi_df, flag_fields)
    dsgc_cont = pick_field(dsgc_df, cont_fields)
    dsgc_flag = pick_field(dsgc_df, flag_fields)

    parasol_lists = rows_to_spike_lists(parasol_df, duration_s, fs, scale_factor=scale_parasol,
                                        integrated_field=parasol_field if parasol_field in cont_fields else None,
                                        flag_field=parasol_field if parasol_field in flag_fields else None)
    midget_lists = rows_to_spike_lists(midget_df, duration_s, fs, scale_factor=scale_midget,
                                       integrated_field=midget_field if midget_field in cont_fields else None,
                                       flag_field=midget_field if midget_field in flag_fields else None)
    smallbi_lists = rows_to_spike_lists(smallbi_df, duration_s, fs, scale_factor=scale_smallbi,
                                        integrated_field=smallbi_field if smallbi_field in cont_fields else None,
                                        flag_field=smallbi_field if smallbi_field in flag_fields else None)

    # DSGC: this script will treat all DSGCs as belonging to a single direction bin (D=1).
    # If you have direction labels, you can bucket them into separate direction lists.
    if len(dsgc_df) > 0:
        if dsgc_cont:
            dsgc_lists = rows_to_spike_lists(dsgc_df, duration_s, fs, scale_factor=5.0,
                                             integrated_field=dsgc_cont)
        elif dsgc_flag:
            dsgc_lists = rows_to_spike_lists(dsgc_df, duration_s, fs, flag_field=dsgc_flag)
        else:
            dsgc_lists = rows_to_spike_lists(dsgc_df, duration_s, fs, scale_factor=5.0,
                                             integrated_field='integrated_signal' if 'integrated_signal' in dsgc_df.columns else None)
    else:
        dsgc_lists = []

    # Build dict expected by VisualCortex.v1.process: keys 'parasol','midget','small_bi', optional '_T_bins'
    T_bins = int(np.ceil(duration_s * fs))
    rgc_dict = {
        'parasol': parasol_lists,
        'midget': midget_lists,
        'small_bi': smallbi_lists,
        '_T_bins': T_bins
    }

    # dsgc_spike_lists expected form: list over directions, each element a list-of-cells arrays.
    # We'll put all DSGCs into a single direction bin if present.
    dsgc_spike_lists = [dsgc_lists] if len(dsgc_lists) > 0 else None

    return rgc_dict, dsgc_spike_lists


def save_outputs(outdir, l4_spikes, v2_spikes, it_spikes):
    os.makedirs(outdir, exist_ok=True)
    # Save L4 counts and first-spike time
    l4_summary = []
    for i, s in enumerate(l4_spikes):
        s = np.asarray(s, dtype=float)
        l4_summary.append({'unit': i, 'n_spikes': s.size, 'first_spike': float(s[0]) if s.size else None})
    pd.DataFrame(l4_summary).to_csv(os.path.join(outdir, 'l4_summary.csv'), index=False)

    # V2 summary
    v2_summary = []
    for i, s in enumerate(v2_spikes):
        s = np.asarray(s, dtype=float)
        v2_summary.append({'unit': i, 'n_spikes': s.size, 'first_spike': float(s[0]) if s.size else None})
    pd.DataFrame(v2_summary).to_csv(os.path.join(outdir, 'v2_summary.csv'), index=False)

    # IT summary
    it_summary = []
    for i, s in enumerate(it_spikes):
        s = np.asarray(s, dtype=float)
        it_summary.append({'unit': i, 'n_spikes': s.size, 'first_spike': float(s[0]) if s.size else None})
    pd.DataFrame(it_summary).to_csv(os.path.join(outdir, 'it_summary.csv'), index=False)

    # Save raw spike lists as npz (could be large)
    np.savez_compressed(os.path.join(outdir, 'spike_outputs.npz'),
                        l4=l4_spikes, v2=v2_spikes, it=it_spikes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parasol', default='parasol.csv')
    p.add_argument('--midget', default='midget.csv')
    p.add_argument('--smallbi', default='small_bi.csv')
    p.add_argument('--dsgc', default='dsgc.csv')
    p.add_argument('--duration', type=float, default=1.0, help='duration in seconds to synthesize')
    p.add_argument('--fs', type=float, default=100.0, help='sampling frequency / frame rate for LGN dt')
    p.add_argument('--outdir', default='vc_demo_out')
    p.add_argument('--scale-parasol', type=float, default=10.0)
    p.add_argument('--scale-midget', type=float, default=5.0)
    p.add_argument('--scale-smallbi', type=float, default=5.0)
    args = p.parse_args()

    # Build synthetic spike lists from CSVs
    rgc_dict, dsgc_spike_lists = build_rgc_dict_from_csv(
        parasol_csv=args.parasol, midget_csv=args.midget, smallbi_csv=args.smallbi, dsgc_csv=args.dsgc,
        duration_s=args.duration, fs=args.fs,
        scale_parasol=args.scale_parasol, scale_midget=args.scale_midget, scale_smallbi=args.scale_smallbi
    )

    print("Constructed RGC dict keys:", list(rgc_dict.keys()))
    if dsgc_spike_lists is not None:
        print("DSGC direction bins:", len(dsgc_spike_lists), "cells in bin:", len(dsgc_spike_lists[0]))

    # Instantiate VisualCortex
    vc = VisualCortex(H=64, W=64, fs=args.fs, n_v1_units=36)

    # Run V1 pipeline with spikes as input (this uses the rgc_spikes_to_frames + v1.process path)
    l4_spikes, feature_maps = vc.v1.process(rgc_spike_dict=rgc_dict, dsgc_spike_lists=dsgc_spike_lists)

    # L2/3, V2, V4, IT orchestration: your VisualCortex.function orchestrates these steps.
    # If you want the full pipeline, call vc.function(...) which expects frames normally.
    # However the VisualCortex.function in your code earlier called self.v1.process internally;
    # so to continue the pipeline we can call vc.function with synthesized frames too.
    # We'll synthesize frames (same helper used earlier) by calling rgc_spikes_to_frames directly if available.
    # prefer using the VisualCortex helper if present
    dt = 1.0 / args.fs
    T_bins = int(np.ceil(args.duration * args.fs))
    try:

        frames = rgc_spikes_to_frames(rgc_dict, T_bins, dt, vc.H, vc.W)
        outputs = vc.function(frames)  # full pipeline orchestration
        l23_spikes = outputs.get('L23', [])
        v2_spikes = outputs.get('V2', [])
        v4_spikes = outputs.get('V4', [])
        it_spikes = outputs.get('IT', [])
    except Exception:
        # fallback: we already have l4_spikes and feature_maps; try to at least call V2/V4/IT manually if objects exist
        try:
            # L23 (if you have L23Unit class and v1.fs metadata)
            T_bins = int(feature_maps.shape[0])
            # Create simple L23 units mirroring earlier orchestration (lightweight)
            Nv1 = len(vc.v1.units)
            l23_units = []
            for k in range(Nv1):
                pres = [(k + delta) % Nv1 for delta in (-1, 0, 1)]
                l23 = L23Unit(presyn_indices=pres, neuron_params=vc.v2._default_neuron_params(),
                              syn_params=vc.v2._default_syn_params(), v1_reference=vc.v1, facilitation_scale=1.2)
                l23_units.append(l23)
            l23_spikes = [u.function(l4_spikes, T_bins) for u in l23_units]
        except Exception:
            l23_spikes = []
        try:
            v2_spikes = [u.function(l4_spikes, T_bins) for u in vc.v2.units]
        except Exception:
            v2_spikes = []
        try:
            v4_spikes = vc.v4.function(v2_spikes, np.zeros((feature_maps.shape[0], vc.W)))
        except Exception:
            v4_spikes = []
        try:
            it_spikes = vc.it.function(v4_spikes)
        except Exception:
            it_spikes = []

    # Save summaries
    save_outputs(args.outdir, l4_spikes, v2_spikes, it_spikes)
    print("Wrote outputs to", args.outdir)


if __name__ == '__main__':
    main()
