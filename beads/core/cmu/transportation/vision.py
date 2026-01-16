import argparse
import pickle

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

"""
4 types of Ganglion Cells:
1. Midget Cells - Small receptive fields that support high-acuity vision and color opponency (red–green)
2. Parasol Cells - High sensitivity to motion and contrast, with rapid responses
3. Small Bistratified Cells - These cells are involved in blue–yellow color processing
4. Direction-Selective Ganglion Cells (DSGCs) - Specifically tuned to respond to motion in one direction
"""


class DSGanglion:
    def __init__(self, sac_cluster, lambda_ds=100.0, integration_factor=1.0, threshold=0.5):
        """
        Direction-Selective Ganglion Cell (DSGC) that integrates directional signals from a cluster of SACs.

        Args:
            sac_cluster (list): List of StarburstAmacrine cells (SACs) belonging to this DSGC's receptive field.
            lambda_ds (float): Spatial decay constant for DSGC integration (micrometers).
            integration_factor (float): Scaling factor for integration.
            threshold (float): Threshold for generating a spike.
        """
        self.sac_cluster = sac_cluster
        # Compute DSGC center as the centroid of SAC centers.
        positions = np.array([sac.center for sac in sac_cluster])
        self.center = np.mean(positions, axis=0)
        self.lambda_ds = lambda_ds
        self.integration_factor = integration_factor
        self.threshold = threshold

        # The integrated contribution vector will be stored as a 2D vector.
        self.contribution_vector = np.array([0.0, 0.0])
        self.integrated_signal = 0.0
        self.spikes = []  # Spike history

    def integrate_SACs(self):
        """
        Integrate contributions from each SAC in the DSGC's cluster.

        For each SAC, compute the 2D vector (from its contribution tensor) and weight it by an exponential decay
        based on the distance between the DSGC center and the SAC center.

        The final contribution vector is the sum of all weighted SAC 2D vectors.

        Returns:
            numpy.ndarray: A 2D vector representing the integrated directional signal.
        """
        net_vector = np.array([0.0, 0.0])
        for sac in self.sac_cluster:
            # Get SAC center and compute distance from DSGC center.
            sac_center = sac.center
            distance = np.linalg.norm(sac_center - self.center)
            weight = np.exp(-distance / self.lambda_ds)
            # Get the SAC's 2D contribution vector.
            sac_vector = sac.get_2d_vector()
            net_vector += weight * sac_vector
        self.contribution_vector = net_vector
        # Optionally, apply a nonlinear transformation (e.g., scaling).
        self.integrated_signal = self.integration_factor * np.linalg.norm(net_vector)

    def function(self):
        """
        Generate a spike based on the integrated directional signal.
        Uses a simple threshold: if the integrated signal exceeds the threshold, generate a spike.

        Returns:
            int: 1 if spike generated, 0 otherwise.
        """
        self.integrate_SACs()
        spike_output = 1 if self.integrated_signal > self.threshold else 0
        self.spikes.append(spike_output)
        return spike_output


def cluster_SACs_upto_group_size(sac_list, sac_group_size=5):
    """
    Cluster SACs into groups of approximately 'sac_group_size' based on spatial proximity.

    Args:
        sac_list (list): List of SAC objects, each with a .center attribute (array-like [x, y]).
        sac_group_size (int): Desired number of SACs per cluster.

    Returns:
        list of list: A list of clusters, each cluster is a list of SAC objects.
    """
    unassigned = sac_list.copy()
    clusters = []

    while unassigned:
        # Start a new cluster with the first unassigned SAC
        seed = unassigned.pop(0)
        # Compute distances from seed to all other unassigned SACs
        seed_center = np.array(seed.center)
        dists = [np.linalg.norm(np.array(sac.center) - seed_center) for sac in unassigned]
        # Sort the remaining SACs by distance
        sorted_indices = np.argsort(dists)
        # Take the nearest (sac_group_size - 1) neighbors
        cluster = [seed]
        for idx in sorted_indices[:sac_group_size - 1]:
            cluster.append(unassigned[idx])
        # Remove those neighbors from unassigned
        for sac in cluster[1:]:
            unassigned.remove(sac)
        clusters.append(cluster)

    return clusters


###############################################################################
# DSGC Layer Initialization
###############################################################################

def initialize_DSGCs(starburst_amacrine_cells, sac_group_size=5, lambda_ds=100.0,
                     integration_factor=1.0, threshold=0.5):
    """
    Cluster SACs from the retina into groups of 'sac_group_size' and assign each group to one DSGC.

    Args:
        starburst_amacrine_cells (list): List of StarburstAmacrine cells (SACs) from the retina.
        sac_group_size (int): Number of SACs per DSGC.
        lambda_ds (float): Spatial decay constant (micrometers) for DSGC integration.
        integration_factor (float): Scaling factor for integration.
        threshold (float): DSGC spiking threshold.

    Returns:
        retina: Updated retina object with retina.ganglion_cells set.
    """
    sac_clusters = cluster_SACs_upto_group_size(starburst_amacrine_cells, sac_group_size)

    dsgc_list = []
    for cluster in sac_clusters:
        dsgc = DSGanglion(sac_cluster=cluster, lambda_ds=lambda_ds,
                          integration_factor=integration_factor, threshold=threshold)
        dsgc_list.append(dsgc)

    return dsgc_list


def exponential_decay(distance, lambda_val):
    """Exponential decay weight based on distance (in micrometers)."""
    return np.exp(-distance / lambda_val)


def unit_vector(angle):
    """Return a 2D unit vector for a given angle (radians)."""
    return np.array([np.cos(angle), np.sin(angle)])


###############################################################################
# Midget Ganglion Cell (Parvocellular Pathway)
###############################################################################
class MidgetGanglion:
    def __init__(self, bipolar_cells, lambda_m=30.0, integration_factor=1.0, threshold=0.3):  # lambda_m is in microns
        """
        Midget ganglion cell model supporting high-acuity vision and red–green color opponency.

        Args:
            bipolar_cells (list): List of midget bipolar cell objects. Each bipolar is assumed to have:
                                  - processed_stimulus (scalar or vector for color opponency)
                                  - x, y (position in micrometers)
            lambda_m (float): Spatial decay constant, reflecting the small receptive field (~30 µm).
            integration_factor (float): Scaling factor.
            threshold (float): Threshold for spike generation.
        """
        self.bipolar_cells = bipolar_cells
        positions = np.array([[b.x, b.y] for b in bipolar_cells])
        self.center = np.mean(positions, axis=0)
        self.lambda_m = lambda_m
        self.integration_factor = integration_factor
        self.threshold = threshold

        self.integrated_signal = 0.0
        self.spikes = []

    def integrate(self):
        """Integrate input from midget bipolar cells with spatial weighting."""
        total = 0.0
        for b in self.bipolar_cells:
            pos = np.array([b.x, b.y])
            distance = np.linalg.norm(pos - self.center)
            weight = exponential_decay(distance, self.lambda_m)
            total += weight * b.get_output()
        self.integrated_signal = self.integration_factor * total

    def function(self):
        """
        Generate a spike if the integrated signal exceeds a threshold.
        """
        self.integrate()
        out = 1 if self.integrated_signal > self.threshold else 0
        self.spikes.append(out)
        return out


###############################################################################
# Parasol Ganglion Cell (Magnocellular Pathway)
###############################################################################
class ParasolGanglion:
    def __init__(self, bipolar_cells, lambda_p=80.0, integration_factor=1.0, threshold=0.4):  # lambda_p is in microns
        """
        Parasol ganglion cell model sensitive to motion and contrast.

        Args:
            bipolar_cells (list): List of diffuse bipolar cell objects (providing large receptive field input).
            lambda_p (float): Spatial decay constant (larger receptive field, e.g., ~80 µm).
            integration_factor (float): Scaling factor.
            threshold (float): Firing threshold.
        """
        self.bipolar_cells = bipolar_cells
        positions = np.array([[b.x, b.y] for b in bipolar_cells])
        self.center = np.mean(positions, axis=0)
        self.lambda_p = lambda_p
        self.integration_factor = integration_factor
        self.threshold = threshold

        self.integrated_signal = 0.0
        self.spikes = []

    def integrate(self):
        """Integrate input from diffuse bipolar cells with spatial weighting."""
        total = 0.0
        for b in self.bipolar_cells:
            pos = np.array([b.x, b.y])
            distance = np.linalg.norm(pos - self.center)
            weight = exponential_decay(distance, self.lambda_p)
            total += weight * b.get_output()
        self.integrated_signal = self.integration_factor * total
        return self.integrated_signal

    def function(self):
        """Generate a spike if integrated signal exceeds threshold."""
        out = 1 if self.integrated_signal > self.threshold else 0
        self.spikes.append(out)
        return out


###############################################################################
# Small Bistratified Ganglion Cell (Koniocellular Pathway)
###############################################################################
class SmallBistratifiedGanglion:
    def __init__(self, bipolar_cells, lambda_sb=60.0, integration_factor=1.0, threshold=0.4):  # lambda_sb is in microns
        """
        Small bistratified ganglion cell model involved in blue–yellow color opponency.

        Args:
            bipolar_cells (list): List of diffuse bipolar cell objects (providing large receptive field input).
            lambda_sb (float): Spatial decay constant for integration (e.g., ~60 µm).
            integration_factor (float): Scaling factor.
            threshold (float): Spike threshold.
        """
        self.bipolar_cells = bipolar_cells
        positions = np.array([[b.x, b.y] for b in bipolar_cells])
        self.center = np.mean(positions, axis=0)
        self.lambda_sb = lambda_sb
        self.integration_factor = integration_factor
        self.threshold = threshold

        self.integrated_signal = 0.0
        self.spikes = []

    def integrate(self):
        """Integrate input from diffuse bipolar cells with spatial weighting."""
        total = 0.0
        for b in self.bipolar_cells:
            pos = np.array([b.x, b.y])
            distance = np.linalg.norm(pos - self.center)
            weight = exponential_decay(distance, self.lambda_sb)
            total += weight * b.get_output()
        self.integrated_signal = self.integration_factor * total
        return self.integrated_signal

    def function(self):
        """Generate a spike if integrated signal exceeds threshold."""
        self.integrate()
        out = 1 if self.integrated_signal > self.threshold else 0
        self.spikes.append(out)
        return out


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


def initialize_midget_cells(cone_bipolar_cells, group_size=8, lambda_m=30.0, integration_factor=1.0, threshold=0.3):
    """
    Cluster r-g cone bipolar cells into groups and create Midget Ganglion Cells.
    """
    # Group midget bipolar cells.
    clusters = cluster_bipolar_upto_group_size(cone_bipolar_cells, group_size, subtypes=['L'])
    midget_cells = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        mg = MidgetGanglion(cluster, lambda_m=lambda_m, integration_factor=integration_factor, threshold=threshold)
        midget_cells.append(mg)
    return midget_cells


def initialize_parasol_cells(cone_bipolar_cells, group_size=8, lambda_p=80.0, integration_factor=1.0, threshold=0.4):
    """
    Cluster diffuse bipolar cells into groups and create Parasol Ganglion Cells.
    """
    # Group diffuse bipolar cells.
    clusters = cluster_bipolar_upto_group_size(cone_bipolar_cells, group_size, subtypes=['', 'M'])
    parasol_cells = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        pg = ParasolGanglion(cluster, lambda_p=lambda_p, integration_factor=integration_factor, threshold=threshold)
        pg.integrate()
        parasol_cells.append(pg)
    return parasol_cells


def initialize_small_bistratified_cells(cone_bipolar_cells, group_size=6, lambda_sb=60.0, integration_factor=1.0,
                                        threshold=0.3):
    """
    Cluster b-y cone bipolar cells into groups and create Small Bistratified Ganglion Cells.
    """
    # Group S bipolar cells
    clusters = cluster_bipolar_upto_group_size(cone_bipolar_cells, group_size, subtypes=['S'])
    small_bistratified_ganglion_cells = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        sbg = SmallBistratifiedGanglion(cluster, lambda_sb=lambda_sb,
                                        integration_factor=integration_factor, threshold=threshold)
        small_bistratified_ganglion_cells.append(sbg)
    return small_bistratified_ganglion_cells


# Temporary code block to test these cells. Input and output should be through files (which can be used for the demo)
def test():
    p = argparse.ArgumentParser()
    p.add_argument("--out_csv", default="/Users/akhilreddy/IdeaProjects/beads/out/visual/dsgc_out.csv")
    args = p.parse_args()

    with open('/Users/akhilreddy/IdeaProjects/beads/out/visual/starburst_amacrine.pkl', 'rb') as file:
        starburst_cells = pickle.load(file)
    print("Loaded S Objects")

    """
    midget_cells = initialize_midget_cells(cone_bipolar_cells)

    records = []
    idx = 0
    for c in midget_cells:
        response = c.function()
        records.append({
            "idx": idx,
            "center": c.center,
            "integrated_signal": float(c.integrated_signal),
            "threshold": float(c.threshold),
            "response": response
        })
        idx += 1

    with open('/Users/akhilreddy/IdeaProjects/beads/out/visual/midget.pkl', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(midget_cells, file)

    parasol_cells = initialize_parasol_cells(cone_bipolar_cells)

    records = []
    idx = 0
    for c in parasol_cells:
        response = c.function()
        records.append({
            "idx": idx,
            "center": c.center,
            "integrated_signal": float(c.integrated_signal),
            "threshold": float(c.threshold),
            "response": response
        })
        idx += 1

    with open('/Users/akhilreddy/IdeaProjects/beads/out/visual/parasol.pkl', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(parasol_cells, file)
    
    small_bistratified_cells = initialize_small_bistratified_cells(cone_bipolar_cells)

    records = []
    idx = 0
    for c in small_bistratified_cells:
        response = c.function()
        records.append({
            "idx": idx,
            "center": c.center,
            "integrated_signal": float(c.integrated_signal),
            "threshold": float(c.threshold),
            "response": response
        })
        idx += 1

    with open('/Users/akhilreddy/IdeaProjects/beads/out/visual/small_bistratified.pkl', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(small_bistratified_cells, file)
    """
    dsgc = initialize_DSGCs(starburst_cells)

    records = []
    idx = 0

    for c in dsgc:
        spike_output = c.function()
        records.append({
            "idx": idx,
            "center": c.center,
            "integrated_signal": float(c.integrated_signal),
            "spike_output": float(spike_output),
        })
        idx += 1

    with open('/Users/akhilreddy/IdeaProjects/beads/out/visual/dsgc.pkl', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(dsgc, file)

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote CSV: {args.out_csv}  (n_cells = {len(df)})")


if __name__ == "__main__":
    test()
