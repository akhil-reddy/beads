import argparse
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components

from beads.core.cmu.sequencing.combine.vision import Bipolar, Horizontal


def sigmoid(x, slope):
    """Compute a sigmoid function with the specified slope."""
    return 1.0 / (1.0 + np.exp(-slope * x))


###############################################################################
# Generic (Non-directional) Amacrine Cell Model
###############################################################################

class GenericAmacrine:
    def __init__(self, tau=0.05, V_rest=-65.0, V_threshold=-50.0,
                 gain=1.2, g_max=1.0, sigmoid_slope=0.3):
        """
        Generic amacrine cell model simulating lateral and temporal modulation.

        Parameters:
            tau (float): Membrane time constant (s) (typical ~50 ms).
            V_rest (float): Resting membrane potential (mV).
            V_threshold (float): Threshold potential (mV) for neurotransmitter release.
            gain (float): Gain factor converting bipolar input into depolarizing drive.
            g_max (float): Maximum inhibitory output conductance (arbitrary units).
            sigmoid_slope (float): Slope of the sigmoidal function for transmitter release.
        """
        self.tau = tau
        self.V_rest = V_rest
        self.V_threshold = V_threshold
        self.gain = gain
        self.g_max = g_max
        self.sigmoid_slope = sigmoid_slope

        self.V = V_rest
        self.V_history = []

    def update(self, bipolar_input, dt):
        """
        Update membrane potential via a leaky integrator and compute inhibitory output.

        dV/dt = [ - (V - V_rest) + gain * bipolar_input ] / tau

        Inhibitory output g = g_max * sigmoid(V - V_threshold, sigmoid_slope)
        """
        dV = (-(self.V - self.V_rest) + self.gain * bipolar_input) * (dt / self.tau)
        self.V += dV
        self.V_history.append(self.V)

        g = self.g_max * sigmoid(self.V - self.V_threshold, self.sigmoid_slope)
        return g

    def reset(self):
        self.V = self.V_rest
        self.V_history = []


###############################################################################
# AII Amacrine Cell Model (Rod Pathway)
###############################################################################

class AIIAmacrine:
    def __init__(self, bipolar_cell, tau=0.04, V_rest=-65.0, V_threshold=-52.0,
                 gain=1.2, g_gap_max=1.0, g_inhib_max=0.8, sigmoid_slope=0.4):
        """
        AII amacrine cell model, specialized for scotopic rod signal processing.

        AII cells receive input exclusively from rod bipolar cells and then:
          - Electrically couple to ON cone bipolar cells (linear transfer).
          - Inhibit OFF cone bipolar cells via glycinergic synapses (nonlinear, steep threshold).

        Parameters:
            bipolar_cell (object): Bipolar cell object
            tau (float): Membrane time constant (s) (typically ~40 ms).
            V_rest (float): Resting membrane potential (mV).
            V_threshold (float): Threshold potential (mV) for glycinergic release.
            gain (float): Gain factor converting rod bipolar input.
            g_gap_max (float): Maximum output for electrical coupling.
            g_inhib_max (float): Maximum inhibitory output for glycinergic transmission.
            sigmoid_slope (float): Slope of the sigmoidal nonlinearity.
        """
        self.latest = None
        self.bipolar_cell = bipolar_cell
        if isinstance(bipolar_cell, Bipolar):
            self.x = bipolar_cell.x
            self.y = bipolar_cell.y

        self.tau = tau
        self.V_rest = V_rest
        self.V_threshold = V_threshold
        self.gain = gain
        self.g_gap_max = g_gap_max
        self.g_inhib_max = g_inhib_max
        self.sigmoid_slope = sigmoid_slope

        self.V = V_rest
        self.V_history = []

    def function(self, rod_bipolar_input, dt=0.01):
        """
        Update the AII cell's membrane potential based on rod bipolar input.
        dV/dt = ( - (V - V_rest) + gain * rod_bipolar_input ) / tau

        Returns:
            tuple: (electrical_output, inhibitory_output)
                electrical_output: Linear output for gap-junction coupling.
                inhibitory_output: Glycinergic output (nonlinear, sigmoid).
        """
        dV = (-(self.V - self.V_rest) + self.gain * rod_bipolar_input) * (dt / self.tau)
        self.V += dV
        self.V_history.append(self.V)

        # Electrical output (linear, normalized between 0 and g_gap_max).
        electrical_output = self.g_gap_max * (self.V - self.V_rest) / (self.V_threshold - self.V_rest)
        electrical_output = np.clip(electrical_output, 0, self.g_gap_max)

        # Inhibitory output (glycinergic release, using a steep sigmoid).
        inhibitory_output = self.g_inhib_max * sigmoid(self.V - self.V_threshold, self.sigmoid_slope)

        self.latest = electrical_output

        return electrical_output, inhibitory_output

    def reset(self):
        self.V = self.V_rest
        self.V_history = []


def initialize_aii_amacrine_cells(rod_bipolar_cells, **params):
    """
    Create an amacrine cell layer associated with each bipolar cell in the retina.

    Args:
        rod_bipolar_cells (list): Should contain a list of rod_bipolar_cells.
        params: Parameters for the chosen amacrine cell model.

    Returns:
        amacrine_cells: A list of AII amacrine cells.
    """
    amacrine_cells = []
    for bipolar in rod_bipolar_cells:
        cell = AIIAmacrine(bipolar, **params)
        amacrine_cells.append(cell)
    return amacrine_cells


def exponential_decay(distance, lambda_r):
    """Compute an exponential decay weight based on distance."""
    return np.exp(-distance / lambda_r)


def unit_vector(angle):
    """Return a 2D unit vector for a given angle (radians)."""
    return np.array([np.cos(angle), np.sin(angle)])


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        else:
            self.p[rb] = ra
            if self.r[ra] == self.r[rb]:
                self.r[ra] += 1


def cluster_bipolar_upto_group_size(bipolar_cells, group_size=8, channel_type=None, leafsize=16, subtypes=None):
    """
    Efficient greedy clustering by spatial proximity.

    - bipolar_cells: list of objects with .x and .y
    - group_size: desired max size per cluster (>=1)
    - channel_type: None (all cells) OR a string (attribute name to match) OR a callable(cell)->bool
    - leafsize: cKDTree parameter (tune for your data size)

    Returns: list of clusters (each cluster is a list of bipolar cell objects).
    """
    if subtypes is not None:
        bipolar_cells = [b for b in bipolar_cells if b.subtype in subtypes]

    if group_size <= 1:
        # trivial clusters: each cell is its own cluster
        return [[b] for b in bipolar_cells]

    N = len(bipolar_cells)
    if N == 0:
        return []

    # Build arrays of positions
    pts = np.empty((N, 2), dtype=np.float64)
    for i, b in enumerate(bipolar_cells):
        pts[i, 0] = float(b.x)
        pts[i, 1] = float(b.y)

    # Build an index list of cells to cluster (respecting channel_type if provided)
    if channel_type is None:
        indices = np.arange(N, dtype=int)
    else:
        if callable(channel_type):
            mask = np.fromiter((1 if channel_type(b) else 0 for b in bipolar_cells), dtype=np.bool_, count=N)
        else:
            # treat channel_type as attribute value to match (attribute name 'channel' or 'channel_type' fallback)
            def _matches(b):
                # try common attribute names
                val = getattr(b, 'channel_type', None)
                if val is None:
                    val = getattr(b, 'channel', None)
                return val == channel_type
            mask = np.fromiter((1 if _matches(b) else 0 for b in bipolar_cells), dtype=np.bool_, count=N)

        indices = np.nonzero(mask)[0]

    M = len(indices)
    if M == 0:
        return []

    sub_pts = pts[indices]

    # k for initial neighbor lookup: at least group_size (include self)
    k0 = min(group_size, M)

    tree = KDTree(sub_pts, leafsize=leafsize)

    # precompute k-nearest neighbors for each candidate (indices are into sub_pts)
    # result shape: (M, k0)
    _, neighbors = tree.query(sub_pts, k=k0, workers=-1)

    # normalize neighbors to 2D array (if k0 == 1, query returns shape (M,), so force 2D)
    neighbors = np.atleast_2d(neighbors)

    assigned = np.zeros(M, dtype=bool)
    clusters = []

    # iterate through candidates in index order (greedy)
    for i_sub in range(M):
        if assigned[i_sub]:
            continue

        # choose seed i_sub and collect up to group_size-1 nearest unassigned neighbors
        chosen = []
        for j in neighbors[i_sub]:
            j = int(j)
            if j == i_sub:
                continue
            if not assigned[j]:
                chosen.append(j)
                if len(chosen) >= group_size - 1:
                    break

        # if not enough neighbors found (because many were assigned), query more neighbors on-the-fly
        if len(chosen) < group_size - 1 and M > k0:
            # ask for a larger neighborhood (cap to M)
            k_more = min(M, max(k0 * 4, group_size * 4))
            _, neigh2 = tree.query(sub_pts[i_sub], k=k_more)
            for j in np.atleast_1d(neigh2):
                j = int(j)
                if j == i_sub or assigned[j] or j in chosen:
                    continue
                chosen.append(j)
                if len(chosen) >= group_size - 1:
                    break

        # form cluster (map sub-indices back to original bipolar_cells indices)
        cluster_sub_indices = [i_sub] + chosen
        orig_indices = indices[cluster_sub_indices].tolist()
        cluster = [bipolar_cells[idx] for idx in orig_indices]
        clusters.append(cluster)

        # mark assigned
        assigned[cluster_sub_indices] = True

    return clusters


###############################################################################
# Starburst Amacrine Cell Model (Directional, with radial grouping)
###############################################################################

class StarburstAmacrine:
    def __init__(self, bipolar_cells, lambda_r=50.0, nonlin_gain=10.0, nonlin_thresh=0.2,
                 tau=0.05):  # lambda is in microns
        """
        Starburst amacrine cell model that integrates inputs from a radially defined focus area.

        Bipolar cells are assumed to be distributed around the SAC's soma.
        Their contributions (weighted by both their output and an exponential decay based on distance)
        are stored separately in a 4D tensor for later processing.

        Args:
            bipolar_cells (list): List of bipolar cell objects, each with:
                                  - processed_stimulus (scalar, 0–1)
                                  - x, y (position in micrometers)
            lambda_r (float): Spatial decay constant (micrometers).
            nonlin_gain (float): Gain for the nonlinear transformation.
            nonlin_thresh (float): Threshold for the nonlinearity.
            tau (float): Time constant for any leaky integration (s); here used for future extension.
        """
        self.bipolar_cells = bipolar_cells
        # Compute focus area as the centroid of the bipolar cell positions.
        positions = np.array([[b.x, b.y] for b in bipolar_cells])
        self.center = np.mean(positions, axis=0)
        self.lambda_r = lambda_r
        self.nonlin_gain = nonlin_gain
        self.nonlin_thresh = nonlin_thresh
        self.tau = tau

        # Instead of a single net vector, we store each bipolar cell's contribution.
        # We'll create a 4D tensor with dimensions:
        # [1, number_of_bipolar_cells, 1, 2] where the last dimension represents (x, y) components.
        self.contribution_tensor = None
        self.preferred_direction = None  # (radians) from the net 2D vector
        self.centrifugal_output = 0.0

    def compute_directional_contributions(self):
        """
        Compute contributions from each bipolar cell by projecting its weighted input onto the 4 cardinal axes.
        The four values correspond to: [pos_x, neg_x, pos_y, neg_y].

        Returns:
            numpy.ndarray: A 1x4 vector with the summed contributions.
        """
        contributions = []
        for b in self.bipolar_cells:
            bipolar_pos = np.array([b.x, b.y])
            r_vec = bipolar_pos - self.center
            r = np.linalg.norm(r_vec)
            if r == 0:
                contributions.append(np.zeros(4))
                continue
            # Weight is bipolar output times an exponential decay based on distance.
            weight = b.get_output() * exponential_decay(r, self.lambda_r)
            unit_r_vec = r_vec / r
            # Project onto 4 cardinal directions.
            proj = np.array([
                max(weight * unit_r_vec[0], 0),  # Positive x
                max(-weight * unit_r_vec[0], 0),  # Negative x
                max(weight * unit_r_vec[1], 0),  # Positive y
                max(-weight * unit_r_vec[1], 0)  # Negative y
            ])
            contributions.append(proj)
        # Sum over all bipolar cells to get a single 1x4 vector.
        summed_contributions = np.sum(np.array(contributions), axis=0)
        self.contribution_tensor = summed_contributions.reshape(1, 4)
        return self.contribution_tensor

    def get_2d_vector(self):
        """
        Convert the 1x4 contribution tensor into a 2D vector.
        The x-component is (pos_x - neg_x) and the y-component is (pos_y - neg_y).

        Returns:
            numpy.ndarray: A 1x2 vector representing the net directional contribution.
        """
        if self.contribution_tensor is None:
            self.compute_directional_contributions()
        proj = self.contribution_tensor.flatten()  # shape (4,)
        x_comp = proj[0] - proj[1]
        y_comp = proj[2] - proj[3]
        net_vector = np.array([x_comp, y_comp])
        mag = np.linalg.norm(net_vector)
        if mag > 1e-3:
            self.preferred_direction = np.arctan2(y_comp, x_comp)
        else:
            self.preferred_direction = None
        return net_vector

    def compute_centrifugal_output(self):
        """
        Compute a nonlinear (sigmoidal) transformation of the net 2D vector's magnitude
        to represent the SAC's directional (centrifugal) output.

        Returns:
            float: Centrifugal output (0–1).
        """
        net_vector = self.get_2d_vector()
        mag = np.linalg.norm(net_vector)
        self.centrifugal_output = 1.0 / (1.0 + np.exp(-self.nonlin_gain * (mag - self.nonlin_thresh)))
        return self.centrifugal_output

    def function(self):
        """
        Update the SAC's state. Here we compute the contributions, derive the 2D vector,
        and then calculate the centrifugal output.

        Returns:
            tuple: (preferred_direction, centrifugal_output, 2D vector)
        """
        self.compute_directional_contributions()
        net_vector = self.get_2d_vector()
        pref_dir = self.preferred_direction
        centrif = self.compute_centrifugal_output()
        return pref_dir, centrif, net_vector


###############################################################################
# Initialization Function for Starburst Amacrine Layer (with Radial Grouping)
###############################################################################

def initialize_starburst_amacrine_cells(cone_bipolar_cells, lambda_r=50.0, nonlin_gain=10.0, nonlin_thresh=0.2, tau=0.05):
    """
    Group cone bipolar cells based on their (x,y) coordinates into clusters (focus areas).
    Create a starburst amacrine cell for each cluster.

    Args:
        cone_bipolar_cells (list): Cone bipolar cells with attributes: x, y, processed_stimulus
        lambda_r: Parameters for the SAC model.
        nonlin_gain: Parameters for the SAC model.
        nonlin_thresh: Parameters for the SAC model.
        tau: Parameters for the SAC model.

    Returns:
       starburst_cells: A list of starburst amacrine cells.
    """
    bipolar_cells = [b for b in cone_bipolar_cells if b.subtype in ['', 'M']]
    # TODO: 356 >= group_size > 1568
    clusters = cluster_bipolar_upto_group_size(bipolar_cells, group_size=356)  # some napkin math for estimating a 50 micron radius
    starburst_cells = []
    for cluster in clusters:
        sac = StarburstAmacrine(cluster, lambda_r=lambda_r, nonlin_gain=nonlin_gain,
                                nonlin_thresh=nonlin_thresh, tau=tau)
        starburst_cells.append(sac)
    return starburst_cells


def initialize_cone_bipolar_cells(horizontal_cells, aii_amacrine_cells):
    """
    Given a retina object that contains a list of horizontal cells with already computed inhibitory responses,
    create a bipolar cell layer.

    Args:
        horizontal_cells: A list of horizontal cell objects, each with a computed attribute `inhibited_stimulus`
        (range normalized to [0, 1]).
        aii_amacrine_cells: A list of AII amacrine cells.

    Returns:
        cone_bipolar_cells: The cone bipolar cells
    """
    zipped_cells = []
    cone_bipolar_cells = []

    for h_cell in horizontal_cells:
        # Create an ON bipolar cell with typical parameter values
        bipolar = Bipolar(h_cell.x, h_cell.y, cell_type='ON', subtype=h_cell.subtype, threshold=0.5, tau=0.07, gain=3.0,
                          saturation=1.0)
        zipped_cells.append((bipolar, h_cell))
        cone_bipolar_cells.append(bipolar)

    print("Finished Horizontal section")

    for aii_cell in aii_amacrine_cells:
        # Create an ON bipolar cell for AII amacrine
        bipolar = Bipolar(aii_cell.x, aii_cell.y, cell_type='ON', threshold=0.5, tau=0.07, gain=3.0, saturation=1.0)
        zipped_cells.append((bipolar, aii_cell))
        cone_bipolar_cells.append(bipolar)

    print("Finished AII section")

    return zipped_cells, cone_bipolar_cells


def deserialize_horizontal_cells(in_path: str):
    """
    Load the serialized structure and recreate Horizontal objects (with pointers).
    Requires Horizontal to be available in scope.
    """
    with open(in_path, 'rb') as f:
        data = pickle.load(f)

    horizontals = [
        Horizontal(d['x'], d['y'], d['subtype'], photoreceptor_cell=d.get('photoreceptor_cell', None),
                   latest=d.get("latest")) for d in data
    ]

    return horizontals


class RemappingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # remap classes pickled under "__main__" to the current module path
        if module == "__main__" and name == "AIIAmacrine":
            module = "beads.core.cmu.sequencing.transforms.vision"
        return super().find_class(module, name)


# Temporary code block to test these cells. Input and output should be through files (which can be used for the demo)
def test():
    p = argparse.ArgumentParser()
    p.add_argument("--out_csv", default="/Users/akhilreddy/IdeaProjects/beads/out/visual/starburst_out.csv")
    args = p.parse_args()

    with open('/Users/akhilreddy/IdeaProjects/beads/out/visual/cone_bipolar.pkl', 'rb') as file:
        cone_bipolar_cells = pickle.load(file)

    starburst_cells = initialize_starburst_amacrine_cells(cone_bipolar_cells)

    records = []
    idx = 0
    for s in starburst_cells:
        response = s.function()
        records.append({
            "idx": idx,
            "preferred_direction": response[0],
            "centrifugal_output": response[1],
            "net_vector": response[2]
        })
        idx += 1

    with open('/Users/akhilreddy/IdeaProjects/beads/out/visual/starburst_amacrine.pkl', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(starburst_cells, file)

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote CSV: {args.out_csv}  (n_cells = {len(df)})")

    """

    horizontal_cells = deserialize_horizontal_cells('/Users/akhilreddy/IdeaProjects/beads/out/visual/horizontal.pkl')
    print("Loaded H Objects")
    with open('/Users/akhilreddy/IdeaProjects/beads/out/visual/aii_amacrine.pkl', 'rb') as file:
        aii_amacrine_cells = RemappingUnpickler(file).load()
    print("Loaded A Objects")

    zipped_cells, cone_bipolar_cells = initialize_cone_bipolar_cells(horizontal_cells, aii_amacrine_cells)

    records = []
    idx = 0
    for cb, c in zipped_cells:
        response = cb.function(c.latest)
        records.append({
            "idx": idx,
            "x_micron": float(cb.x),
            "y_micron": float(cb.y),
            "subtype": cb.subtype,
            "response": response
        })
        idx += 1

    with open('/Users/akhilreddy/IdeaProjects/beads/out/visual/cone_bipolar.pkl', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(cone_bipolar_cells, file)

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote CSV: {args.out_csv}  (n_cells = {len(df)})")

    with open('/Users/akhilreddy/IdeaProjects/beads/out/visual/rod_bipolar.pkl', 'rb') as file:
        rod_bipolar_cells = pickle.load(file)

    aii_amacrine_cells = initialize_aii_amacrine_cells(rod_bipolar_cells)

    records = []
    idx = 0
    for a, r in zip(aii_amacrine_cells, rod_bipolar_cells):
        response = a.function(r.get_output())
        records.append({
            "idx": idx,
            "V": float(a.V),
            "electrical_output": response[0],
            "inhibitory_output": response[1],
        })
        idx += 1

    with open('/Users/akhilreddy/IdeaProjects/beads/out/visual/aii_amacrine.pkl', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(aii_amacrine_cells, file)
        
    df = pd.DataFrame.from_records(records)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote CSV: {args.out_csv}  (n_cells = {len(df)})")
    """


if __name__ == "__main__":
    test()
