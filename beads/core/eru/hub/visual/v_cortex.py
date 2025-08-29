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
# V2, V4, IT (simple pooling/aggregation classes)
# ---------------------------
class V2:
    def __init__(self, v1_obj, pool=(2, 2)):
        self.v1 = v1_obj
        self.pool = pool

    def process(self, video_gray):
        # naive pooling of V1 spike counts into V2 units
        v1_spikes = self.v1.process(video_gray)  # list of arrays
        # sum rates across local neighborhoods to produce V2 "responses"
        # return list of spike arrays (placeholder simple mapping)
        v2_out = []
        for i, s in enumerate(v1_spikes):
            # convert spike times to rates and threshold to pseudo-spikes
            rate = len(s) / max(1, video_gray.shape[0] * self.v1.strf.temporal.size / self.v1.fs)
            if rate > 0.1:
                v2_out.append(s)
            else:
                v2_out.append(np.array([]))
        return v2_out


class V4:
    def __init__(self, v2_obj):
        self.v2 = v2_obj

    def process(self, video_gray):
        v2_out = self.v2.process(video_gray)
        # simple pooling across v2 to produce curvature/color sensitive pseudosignals
        v4_out = []
        for i, s in enumerate(v2_out):
            if len(s) > 0:
                v4_out.append(s)
            else:
                v4_out.append(np.array([]))
        return v4_out


class IT:
    def __init__(self, v4_obj, n_units=32):
        self.v4 = v4_obj
        self.n_units = n_units
        # simple random projections (placeholder for associative memory)
        self.proj = [np.random.randn(100) for _ in range(n_units)]

    def process(self, video_gray):
        v4_out = self.v4.process(video_gray)
        it_out = []
        for i in range(self.n_units):
            # simple readout: if any v4 unit spiked, mark IT unit as spiking
            any_spike = any(len(s) > 0 for s in v4_out)
            if any_spike:
                it_out.append(np.array([0.0]))  # single spike at t=0 (placeholder)
            else:
                it_out.append(np.array([]))
        return it_out


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
