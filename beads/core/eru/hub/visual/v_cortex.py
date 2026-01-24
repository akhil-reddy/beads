import argparse
import os
import pickle
import numpy as np
from scipy import sparse
from scipy.signal import lfilter
from scipy.spatial import KDTree

from beads.core.eru.interneuron import ShortTermSynapse, MultiCompartmentNeuron, Receptor


# TODO: Verify this simplification
# -----------------------------------------------------------------------------
# 1. HELPER: VECTORIZED CONNECTIVITY
# -----------------------------------------------------------------------------
def generate_weight_matrix(src_locs, dst_locs, func_type, **kwargs):
    """
    Generates sparse weight matrices for connections between layers
    using spatial indexing (cKDTree) for O(N) memory efficiency.
    """
    if len(src_locs) == 0 or len(dst_locs) == 0:
        return None

    # 1. Determine Search Radius
    # We analytically calculate the distance where weights naturally drop
    # below the threshold (0.005), so we don't compute useless connections.
    threshold = 0.005
    radius_factor = 3.5  # approx sqrt(-2 * ln(0.005))

    if func_type == 'DoG':
        sigma_c = kwargs.get('sigma_c', 1.0)
        sigma_s = kwargs.get('sigma_s', 3.0)
        # The wider surround sigma dictates the reach
        search_radius = max(sigma_c, sigma_s) * radius_factor

    elif func_type == 'Gabor':
        sigma = kwargs.get('sigma', 2.0)
        search_radius = sigma * radius_factor

    elif func_type == 'Gaussian':
        sigma = kwargs.get('sigma', 2.0)
        search_radius = sigma * radius_factor
    else:
        raise ValueError(f"Unknown func_type: {func_type}")

    # 2. Efficient Spatial Search
    # Build trees for fast nearest-neighbor lookup
    tree_src = KDTree(src_locs)
    tree_dst = KDTree(dst_locs)

    # sparse_distance_matrix returns a COO-style sparse matrix of distances
    # only for pairs within search_radius.
    dist_mat = tree_dst.sparse_distance_matrix(
        tree_src,
        max_distance=search_radius,
        output_type='coo_matrix'
    )

    if dist_mat.nnz == 0:
        return sparse.csr_matrix((len(dst_locs), len(src_locs)))

    # 3. Vectorized Weight Calculation on Sparse Data
    rows = dist_mat.row
    cols = dist_mat.col
    dist_sq = dist_mat.data ** 2

    weights = np.zeros_like(dist_sq)

    if func_type == 'DoG':
        # Explicitly extract parameters here for clarity/safety
        sigma_c = kwargs.get('sigma_c', 1.0)
        sigma_s = kwargs.get('sigma_s', 3.0)

        t1 = np.exp(-dist_sq / (2 * sigma_c ** 2))
        t2 = 0.5 * np.exp(-dist_sq / (2 * sigma_s ** 2))
        weights = t1 - t2

    elif func_type == 'Gaussian':
        sigma = kwargs.get('sigma', 2.0)
        weights = np.exp(-dist_sq / (2 * sigma ** 2))

    elif func_type == 'Gabor':
        # Explicitly extract parameters
        sigma = kwargs.get('sigma', 2.0)
        theta = kwargs.get('theta', 0.0)
        lam = kwargs.get('lambda', 5.0)  # 'lambda' is a reserved keyword, usually 'lam' is safer

        # Calculate Oriented Distances
        dx = dst_locs[rows, 0] - src_locs[cols, 0]
        dy = dst_locs[rows, 1] - src_locs[cols, 1]

        x_th = dx * np.cos(theta) + dy * np.sin(theta)
        weights = np.exp(-dist_sq / (2 * sigma ** 2)) * np.cos(2 * np.pi * x_th / lam)

    # 4. Final Thresholding
    # Even within radius, Gabor/DoG zero-crossings might produce near-zero weights
    mask = np.abs(weights) >= threshold

    # Construct CSR matrix
    W = sparse.csr_matrix(
        (weights[mask], (rows[mask], cols[mask])),
        shape=(len(dst_locs), len(src_locs))
    )

    return W


def bin_spikes(spike_lists, T_bins, dt):
    """Convert discrete spike times into a sparse rate matrix (N_neurons, T_bins)."""
    if not spike_lists: return None
    N = len(spike_lists)
    mat = np.zeros((N, T_bins), dtype=np.float32)
    for i, times in enumerate(spike_lists):
        if len(times) == 0: continue
        bins = (np.array(times) / dt).astype(int)
        bins = bins[(bins >= 0) & (bins < T_bins)]
        np.add.at(mat[i], bins, 1.0)
    return mat


# -----------------------------------------------------------------------------
# 2. DATA ADAPTOR (Reads your .pkl files)
# -----------------------------------------------------------------------------
class RetinalDataAdaptor:
    @staticmethod
    def extract_from_pkl(path, dt):
        if not path or not os.path.exists(path):
            print(f"  [Warn] File not found: {path}")
            return np.empty((0, 2)), []

        with open(path, 'rb') as f:
            cells = pickle.load(f)

        coords, spikes = [], []
        for c in cells:
            # Handle different center formats
            cen = np.array(c.center).flatten()
            coords.append(cen[:2])

            # Convert binary spike history to times
            raw = np.array(c.spikes)
            times = np.where(raw > 0)[0] * dt
            spikes.append(times)

        return np.array(coords), spikes


# -----------------------------------------------------------------------------
# 3. LGN LAYER (Magno, Parvo, Konio)
# -----------------------------------------------------------------------------
class LGNLayer:
    def __init__(self, h, w, channel, neuron_params):
        self.n = h * w
        self.channel = channel  # 'M', 'P', 'K'
        self.locs = np.column_stack([np.linspace(0, h, self.n) % h, np.linspace(0, w, self.n) // h])
        self.units = []
        self.weights = {}  # Dict to store weights from multiple sources (e.g. Parasol, DSGC)

        for _ in range(self.n):
            neu = MultiCompartmentNeuron(neuron_params.copy())
            neu.add_receptor(Receptor(g_max=1.5e-9, E_rev=0.0, tau_rise=0.002, tau_decay=0.002, location='soma'))
            self.units.append(neu)

    def connect(self, rgc_locs, rgc_name):
        if len(rgc_locs) == 0: return
        print(f"  > Wiring {rgc_name} -> LGN {self.channel}")

        # Select filter params based on pathway
        if self.channel == 'M':
            params = {'sigma_c': 1.5, 'sigma_s': 5.0}  # Wide
        elif self.channel == 'P':
            params = {'sigma_c': 0.5, 'sigma_s': 1.5}  # Narrow
        else:
            params = {'sigma_c': 1.0, 'sigma_s': 3.0}  # Konio

        self.weights[rgc_name] = generate_weight_matrix(rgc_locs, self.locs, 'DoG', **params)

    def function(self, inputs_dict, feedback_gain=None):
        """
        inputs_dict: { 'parasol': mat, 'dsgc': mat, ... }
        """
        total_drive = np.zeros((self.n, list(inputs_dict.values())[0].shape[1]))

        # 1. Sum Synaptic Inputs
        for name, mat in inputs_dict.items():
            if name in self.weights and mat is not None:
                total_drive += self.weights[name].dot(mat)

        # 2. Apply Corticothalamic Feedback (L5/6 Gain)
        if feedback_gain is not None:
            total_drive *= feedback_gain[:, np.newaxis]

        # 3. Temporal Filter (Magno=Transient, Parvo=Sustained)
        if self.channel == 'M':
            total_drive = lfilter([1, -1], [1, -0.9], total_drive, axis=1) * 5.0
        elif self.channel == 'P':
            total_drive = lfilter([0.2], [1, -0.8], total_drive, axis=1) * 2.0

        # 4. Integrate Neurons
        out = []
        for i, u in enumerate(self.units):
            inp = np.maximum(total_drive[i], 0)[:, np.newaxis]
            times, _ = u.step(inp)
            out.append(times)
        return out


# -----------------------------------------------------------------------------
# 4. V1 LAYERS (L4, L2/3, L5/6)
# -----------------------------------------------------------------------------
class V1L4:
    """Input Layer: Gabor Filtering."""

    def __init__(self, h, w, n_or, params, syn_params):
        self.n = h * w * n_or
        self.n_or = n_or
        # Locs repeated for each orientation
        grid = np.column_stack([np.linspace(0, h, h * w) % h, np.linspace(0, w, h * w) // h])
        self.locs = np.repeat(grid, n_or, axis=0)
        self.thetas = np.tile(np.linspace(0, np.pi, n_or), h * w)

        self.w_M, self.w_P, self.w_K = None, None, None
        self.units = []
        for _ in range(self.n):
            neu = MultiCompartmentNeuron(params.copy())
            neu.add_receptor(Receptor(g_max=1e-9, E_rev=0.0, location='soma'))
            syn = ShortTermSynapse(**syn_params)
            self.units.append({'neu': neu, 'syn': syn})

    def connect(self, locs_M, locs_P, locs_K):
        # M -> V1 (Dominant motion/contrast)
        self.w_M = generate_weight_matrix(locs_M, self.locs, 'Gabor', theta=self.thetas[:, None], sigma=2.5)
        # P -> V1 (Detail)
        self.w_P = generate_weight_matrix(locs_P, self.locs, 'Gabor', theta=self.thetas[:, None], sigma=1.2)
        # K -> V1 (Blobs/Color)
        self.w_K = generate_weight_matrix(locs_K, self.locs, 'Gaussian', sigma=3.0)

    def function(self, sM, sP, sK):
        drive = np.zeros((self.n, sM.shape[1]))
        if self.w_M is not None: drive += self.w_M.dot(sM) * 1.0
        if self.w_P is not None: drive += self.w_P.dot(sP) * 0.8
        if self.w_K is not None: drive += self.w_K.dot(sK) * 0.5

        out = []
        for i, u in enumerate(self.units):
            # STP + Neuron
            norm = np.clip(drive[i], 0, 50) / 50.0
            syn_d = u['syn'].step(norm) * np.max(drive[i])
            times, _ = u['neu'].step(np.maximum(syn_d, 0)[:, np.newaxis])
            out.append(times)
        return out


class V1L23:
    """Horizontal Layer: Lateral Facilitation."""

    def __init__(self, l4, params):
        self.n = l4.n
        self.locs = l4.locs
        # Lateral weights (Gaussian) for context
        self.w_lat = generate_weight_matrix(self.locs, self.locs, 'Gaussian', sigma=3.0)
        self.units = [MultiCompartmentNeuron(params.copy()) for _ in range(self.n)]
        for u in self.units: u.add_receptor(Receptor(g_max=1.5e-9, E_rev=0.0, tau_rise=0.002, tau_decay=0.002, location='soma'))

    def function(self, l4_mat):
        # 1. Feedforward
        drive = l4_mat.copy()
        # 2. Lateral Context (Sum of neighbors)
        context = self.w_lat.dot(l4_mat)
        # 3. Facilitation (Gain boost if neighbors active)
        gain = 1.0 + 0.5 * (context > 0.5)
        total = drive * gain

        out = []
        for i, u in enumerate(self.units):
            times, _ = u.step(total[i, :, None] * 10e-9)
            out.append(times)
        return out


class V1L56:
    """Feedback Layer: Computes Gain Map for LGN."""

    def __init__(self, l23, h, w):
        self.target_locs = np.column_stack([np.linspace(0, h, h * w) % h, np.linspace(0, w, h * w) // h])
        # Pools L2/3 activity back to retinotopic map
        self.w_pool = generate_weight_matrix(l23.locs, self.target_locs, 'Gaussian', sigma=5.0)

    def compute_feedback(self, l23_mat):
        rates = np.sum(l23_mat, axis=1)
        map_act = self.w_pool.dot(rates)
        # Normalize to gain 1.0 - 1.5
        mx = np.max(map_act) + 1e-9
        return 1.0 + (map_act / mx) * 0.5


# -----------------------------------------------------------------------------
# 5. HIGHER CORTEX (V2, V4, IT)
# -----------------------------------------------------------------------------
class V2Layer:
    """Corner Detection (AND logic on V1 orientations)."""

    def __init__(self, v1, n_units, params):
        self.units = []
        rng = np.random.default_rng(42)
        n_v1 = v1.n

        for _ in range(n_units):
            # Connect to 2 V1 units (likely different orientations)
            ids = rng.choice(n_v1, 2, replace=False)
            neu = MultiCompartmentNeuron(params.copy())
            neu.add_receptor(Receptor(g_max=1e-9, E_rev=0.0, tau_rise=0.002, tau_decay=0.002, location='soma'))
            self.units.append({'ids': ids, 'neu': neu})

    def function(self, v1_mat):
        out = []
        for u in self.units:
            # Coincidence Detection (Multiplication / AND)
            in_a = v1_mat[u['ids'][0]]
            in_b = v1_mat[u['ids'][1]]
            drive = (in_a * in_b) * 20e-9  # High gain for coincidence
            times, _ = u['neu'].step(drive[:, None])
            out.append(times)
        return out


class V4Layer:
    """Shape Pooling / Invariance."""

    def __init__(self, v2_units, n_units, params):
        self.units = []
        rng = np.random.default_rng(99)
        n_in = len(v2_units)

        for _ in range(n_units):
            # Pool from multiple V2 units
            ids = rng.choice(n_in, min(8, n_in), replace=False)
            neu = MultiCompartmentNeuron(params.copy())
            neu.add_receptor(Receptor(g_max=1e-9, E_rev=0.0, tau_rise=0.002, tau_decay=0.002, location='soma'))
            self.units.append({'ids': ids, 'neu': neu})

    def function(self, v2_mat):
        out = []
        for u in self.units:
            # Sum pooling + Normalization
            raw = np.sum(v2_mat[u['ids']], axis=0)
            norm = raw / (np.mean(raw) + 1e-9) * 10e-9
            times, _ = u['neu'].step(norm[:, None])
            out.append(times)
        return out


class ITLayer:
    """Object Readout (Prototypes)."""

    def __init__(self, n_inputs, n_classes=10):
        self.protos = np.random.rand(n_classes, n_inputs)
        # Normalize
        self.protos /= (np.linalg.norm(self.protos, axis=1, keepdims=True) + 1e-9)

    def function(self, v4_spikes):
        # Rate coding
        rates = np.array([len(s) for s in v4_spikes])
        norm = np.linalg.norm(rates) + 1e-9
        vec = rates / norm
        scores = self.protos.dot(vec)
        return scores


# -----------------------------------------------------------------------------
# 6. ORCHESTRATOR
# -----------------------------------------------------------------------------
class VisualCortex:
    def __init__(self, H=64, W=64, fs=100.0):
        self.inputs = None
        self.fs = fs
        self.dt = 1.0 / fs

        # Standard Params
        p_neu = {'C': [200e-12], 'g_L': [10e-9], 'E_L': -65e-3,
                 'g_Na': 1200e-9, 'E_Na': 50e-3, 'g_K': 360e-9, 'E_K': -77e-3, 'dt': self.dt}
        p_syn = {'U': 0.4, 'tau_rec': 0.4, 'dt': self.dt}

        # --- Instantiate Layers ---
        self.lgn_M = LGNLayer(H, W, 'M', p_neu)
        self.lgn_P = LGNLayer(H, W, 'P', p_neu)
        self.lgn_K = LGNLayer(H, W, 'K', p_neu)

        self.v1_l4 = V1L4(H, W, 4, p_neu, p_syn)
        self.v1_l23 = V1L23(self.v1_l4, p_neu)
        self.v1_l56 = V1L56(self.v1_l23, H, W)

        self.v2 = V2Layer(self.v1_l23, 64, p_neu)
        self.v4 = V4Layer(self.v2.units, 32, p_neu)
        self.it = ITLayer(32, 10)

        self.gain_cache = None

    def load_and_connect(self, file_map):
        print("\n[1] Loading Data...")
        self.inputs = {}
        adap = RetinalDataAdaptor()

        for k, path in file_map.items():
            coords, spikes = adap.extract_from_pkl(path, self.dt)
            self.inputs[k] = {'c': coords, 's': spikes}
            print(f" - {k}: {len(spikes)} cells loaded.")

        print("[2] Building Connectivity (Vectorized)...")
        # Wiring Logic:
        # Parasol + DSGC -> Magno
        self.lgn_M.connect(self.inputs['parasol']['c'], 'parasol')
        self.lgn_M.connect(self.inputs['dsgc']['c'], 'dsgc')
        # Midget -> Parvo
        self.lgn_P.connect(self.inputs['midget']['c'], 'midget')
        # Small Bi -> Konio
        self.lgn_K.connect(self.inputs['small_bi']['c'], 'small_bi')

        # LGN -> V1
        self.v1_l4.connect(self.lgn_M.locs, self.lgn_P.locs, self.lgn_K.locs)

    def function(self, duration):
        tb = int(duration * self.fs)
        print(f"\n[3] Running Simulation ({duration}s)...")

        # Bin Inputs
        mats = {k: bin_spikes(v['s'], tb, self.dt) for k, v in self.inputs.items()}

        # --- LGN ---
        print("    > LGN processing...")
        # Note: M layer gets inputs from both Parasol and DSGC
        lgn_m = self.lgn_M.function({'parasol': mats['parasol'], 'dsgc': mats['dsgc']}, self.gain_cache)
        lgn_p = self.lgn_P.function({'midget': mats['midget']}, self.gain_cache)
        lgn_k = self.lgn_K.function({'small_bi': mats['small_bi']}, self.gain_cache)

        # --- V1 ---
        print("    > V1 (L4 -> L2/3 -> L5/6)...")
        l4_out = self.v1_l4.function(bin_spikes(lgn_m, tb, self.dt),
                                    bin_spikes(lgn_p, tb, self.dt),
                                    bin_spikes(lgn_k, tb, self.dt))

        l23_out = self.v1_l23.function(bin_spikes(l4_out, tb, self.dt))

        # Compute Feedback Gain for next loop (or save for analysis)
        self.gain_cache = self.v1_l56.compute_feedback(bin_spikes(l23_out, tb, self.dt))

        # --- Higher Areas ---
        print("    > Higher Areas (V2 -> V4 -> IT)...")
        v2_out = self.v2.function(bin_spikes(l23_out, tb, self.dt))
        v4_out = self.v4.function(bin_spikes(v2_out, tb, self.dt))
        it_out = self.it.function(v4_out)

        return {
            'LGN_M': lgn_m, 'LGN_P': lgn_p, 'LGN_K': lgn_k,
            'V1_L4': l4_out, 'V1_L23': l23_out, 'V2': v2_out,
            'V4': v4_out, 'IT': it_out, 'Feedback': self.gain_cache
        }


# TODO: Temporary code block to test these cells. Input and output should be through files (which can be used for the demo)
def demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parasol', default='out/visual/parasol.pkl')
    parser.add_argument('--midget', default='out/visual/midget.pkl')
    parser.add_argument('--small_bi', default='out/visual/small_bistratified.pkl')
    parser.add_argument('--dsgc', default='out/visual/dsgc.pkl')
    parser.add_argument('--duration', type=float, default=0.5)
    parser.add_argument('--out', default='cortex_results.npz')
    args = parser.parse_args()

    # Initialize
    vc = VisualCortex(H=64, W=64, fs=100.0)

    # Load
    vc.load_and_connect({
        'parasol': args.parasol,
        'midget': args.midget,
        'small_bi': args.small_bi,
        'dsgc': args.dsgc
    })

    # Run
    res = vc.function(args.duration)

    # Report
    print("\n[4] Results Summary:")
    print(f"    LGN Magno Spikes: {sum(len(s) for s in res['LGN_M'])}")
    print(f"    V1 L2/3 Spikes:   {sum(len(s) for s in res['V1_L23'])}")
    print(f"    V2 Spikes:        {sum(len(s) for s in res['V2'])}")
    print(f"    V4 Spikes:        {sum(len(s) for s in res['V4'])}")
    print(f"    IT Protos Scores: {np.round(res['IT'], 3)}")

    # Save
    np.savez(args.out, **res)
    print(f" Saved full outputs to {args.out}")


if __name__ == "__main__":
    demo()
