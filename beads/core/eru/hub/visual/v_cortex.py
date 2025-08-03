import numpy as np
from scipy.signal import fftconvolve

from beads.core.eru.hub.audio.a1_cortex import ShortTermSynapse, ConductanceLIFNeuron, SpectroTemporalReceptiveField


# ----------------------------------------
# 1. Retinal & LGN Front-End
# ----------------------------------------
class PhotoreceptorArray:
    def __init__(self, sens_curves):
        self.sens = sens_curves

    def apply(self, frames):
        return np.tensordot(frames,
                            np.stack([self.sens['L'], self.sens['M'], self.sens['S']]),
                            axes=([3], [0]))


class CenterSurroundCell:
    def __init__(self, size=9, sigma_c=1.0, sigma_s=3.0):
        xs = np.arange(size) - size // 2
        xv, yv = np.meshgrid(xs, xs)
        self.kernel = (np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma_c ** 2)) -
                       np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma_s ** 2)))
        self.kernel /= np.sum(np.abs(self.kernel))

    def apply(self, img):
        return fftconvolve(img, self.kernel, mode='same')


class LGNRelay:
    def __init__(self, burst_thresh=0.2):
        self.burst_thresh = burst_thresh

    def apply(self, cs_signal):
        return cs_signal > self.burst_thresh


# ----------------------------------------
# 2. 3D STRF as ERU Spatial-Temporal Module
# ----------------------------------------
class STRF3D:
    def __init__(self, spatial_size, spatial_sigma, temporal_taps, orientation,
                 spatial_bandwidth, temporal_sigma, modulation_rate, fs):
        # create spatial ERU-STRF by treating 2D Gabor as separable from 1D temporal
        self.spatial = SpectroTemporalReceptiveField(1, spatial_size, 0, spatial_bandwidth,
                                                     0, spatial_sigma, modulation_rate, fs, 1).spectral.reshape(
            spatial_size, 1)
        # temporal from auditory ERU
        self.temporal = SpectroTemporalReceptiveField(1, temporal_taps, 0, 1,
                                                      temporal_taps // 2, temporal_sigma, modulation_rate, fs,
                                                      1).temporal
        self.spatial = self.spatial / np.linalg.norm(self.spatial)
        self.temporal = self.temporal / np.linalg.norm(self.temporal)

    def apply(self, volume):
        T, H, W = volume.shape
        # spatial conv per frame (orientation omitted for brevity)
        spat_out = np.stack([fftconvolve(volume[t],
                                         self.spatial, mode='same').flatten()
                             for t in range(T)])
        # temporal conv
        temp = fftconvolve(spat_out,
                           self.temporal[:, None],
                           mode='valid')
        return temp.reshape(-1, H, W)


# ----------------------------------------
# 3. Neurons & Synapses for V1/V2/V4/IT
# ----------------------------------------
class VisualUnit:
    def __init__(self, neuron_params, syn_params, strf_params=None):
        self.strf = STRF3D(**strf_params) if strf_params else None
        self.syn = ShortTermSynapse(**syn_params)
        self.neu = ConductanceLIFNeuron(**neuron_params)

    def step(self, input_signal):
        # apply STRF if present
        drive = self.strf.apply(input_signal) if self.strf else input_signal
        # threshold to spikes
        spikes = drive > np.percentile(drive, 90)
        # synaptic current
        I = self.syn.step(spikes.astype(float))
        # neuron dynamics
        out_spikes = []
        for i in range(len(I)):
            if self.neu.step(I[i] > 0): out_spikes.append(i)
        return out_spikes


# ----------------------------------------
# 4. Define Layers and Interneuron Diversity
# ----------------------------------------
def build_visual_layer(num_units, neuron_base, syn_base, strf_base=None, interneuron_types=None):
    units = []
    for _ in range(num_units):
        units.append(VisualUnit(neuron_base, syn_base, strf_base))
    # add interneurons if specified
    if interneuron_types:
        for it in interneuron_types:
            params = neuron_base.copy()
            params.update(it['neuron_override'])
            units.append(VisualUnit(params, syn_base, None))
    return units


# ----------------------------------------
# 5. Visual Cortex Full Integration
# ----------------------------------------
class VisualCortexERU:
    def __init__(self, config):
        # front-end
        self.photoreceptors = PhotoreceptorArray(config['sens_curves'])
        self.center_surround = CenterSurroundCell()
        self.lgn = LGNRelay()
        # Layer-specific parameters
        self.v1 = build_visual_layer(
            num_units=config['v1_count'],
            neuron_base=config['neurons']['pyramidal'],
            syn_base=config['synapses'],
            strf_base=config['v1_strf'],
            interneuron_types=config['v1_interneurons']
        )
        self.v2 = build_visual_layer(
            num_units=config['v2_count'],
            neuron_base=config['neurons']['pyramidal'],
            syn_base=config['synapses'],
            strf_base=None,
            interneuron_types=config['v2_interneurons']
        )
        self.v4 = build_visual_layer(
            num_units=config['v4_count'],
            neuron_base=config['neurons']['pyramidal'],
            syn_base=config['synapses'],
            strf_base=None,
            interneuron_types=config['v4_interneurons']
        )
        self.it = build_visual_layer(
            num_units=config['it_count'],
            neuron_base=config['neurons']['pyramidal'],
            syn_base=config['synapses'],
            strf_base=None,
            interneuron_types=None
        )

    def run(self, frames):
        # retina & LGN
        lms = self.photoreceptors.apply(frames)
        cs = np.stack([self.center_surround.apply(lms[t]) for t in range(frames.shape[0])])
        spikes_lgn = self.lgn.apply(cs)
        # pass through layers
        signals = cs
        outputs = {}
        for name, layer in zip(['v1', 'v2', 'v4', 'it'], [self.v1, self.v2, self.v4, self.it]):
            layer_out = []
            for unit in layer:
                out = unit.step(signals)
                layer_out.append(out)
            outputs[name] = layer_out
            # convert spikes back to continuous for the next stage
            signals = np.array([np.bincount(u, minlength=signals.shape[1] * signals.shape[2])
                               .reshape(signals.shape[1], signals.shape[2])
                                for u in layer_out]).mean(axis=0)
        return outputs
