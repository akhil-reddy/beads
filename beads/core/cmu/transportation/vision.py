import numpy as np

"""
4 types of Ganglion Cells:
1. Midget Cells - Small receptive fields that support high-acuity vision and color opponency (red–green)
2. Parasol Cells - High sensitivity to motion and contrast, with rapid responses
3. Small Bistratified Cells - These cells are involved in blue–yellow color processing
4. Direction-Selective Ganglion Cells (DSGCs) - Specifically tuned to respond to motion in one direction
"""


def cluster_SACs(sac_list, group_size=5):
    """
    Cluster SACs into groups of approximately 'group_size' based on their centers.

    For simplicity, we group them in the order they appear.

    Args:
        sac_list (list): List of SAC objects (each must have a 'center' attribute, a 2D array).
        group_size (int): Number of SACs per DSGC.

    Returns:
        list: A list of clusters, each cluster is a list of SAC objects.
    """
    clusters = []
    current_cluster = []
    for sac in sac_list:
        current_cluster.append(sac)
        if len(current_cluster) >= group_size:
            clusters.append(current_cluster)
            current_cluster = []
    if current_cluster:
        clusters.append(current_cluster)
    return clusters


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

    """
    def update_bipolar_firing_rate(self, max_proj=6.0):  # max_proj is DSGC focus radius in microns
    
        For each SAC in the DSGC's cluster, iterate over its bipolar cells.
        For each bipolar cell, compute the projection of its relative position (to DSGC center)
        onto the DSGC's integrated 2D vector. If this projection is high, increase the bipolar cell's
        firing rate.

        This method simulates feedback where the DSGC modulates the firing rate of bipolar cells
        based on alignment with the integrated directional signal.
        
        # Ensure DSGC contribution vector is updated.
        self.integrate_SACs()
        # Normalize the DSGC integrated vector (if non-zero).
        norm = np.linalg.norm(self.contribution_vector)
        if norm < 1e-3:
            unit_dir = np.array([0.0, 0.0])
        else:
            unit_dir = self.contribution_vector / norm

        for sac in self.sac_cluster:
            for b in sac.bipolar_cells:
                bipolar_pos = np.array([b.x, b.y])
                # Relative vector from DSGC center to bipolar cell.
                rel_vec = bipolar_pos - self.center
                # Projection of this vector onto DSGC's integrated direction.
                projection = np.dot(rel_vec, unit_dir)
                # Normalize the projection.
                normalized_projection = projection / max_proj
                # For demonstration, if normalized projection exceeds a threshold, increase firing rate.
                if not hasattr(b, 'firing_rate'):
                    b.firing_rate = 20.0  # baseline firing rate, in Hz
                # Increase firing rate proportionally to the normalized projection.
                b.firing_rate += normalized_projection  # scaling factor can be adjusted
    """

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


def cluster_SACs_by_proximity(sac_list, sac_group_size=5):
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
    sac_clusters = cluster_SACs_by_proximity(starburst_amacrine_cells, sac_group_size)

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
            total += weight * b.processed_stimulus
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
            total += weight * b.processed_stimulus
        self.integrated_signal = self.integration_factor * total
        return self.integrated_signal

    def spike(self):
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
            total += weight * b.processed_stimulus
        self.integrated_signal = self.integration_factor * total
        return self.integrated_signal

    def spike(self):
        """Generate a spike if integrated signal exceeds threshold."""
        out = 1 if self.integrated_signal > self.threshold else 0
        self.spikes.append(out)
        return out


def cluster_bipolar_by_proximity(bipolar_cells, group_size=8, channel_type=None):
    """
    Cluster bipolar cells into groups of up to 'group_size' based on spatial proximity.

    Args:
        bipolar_cells (list): List of bipolar cell objects, each with .x and .y attributes.
        group_size (int): Desired maximum number of cells per cluster.
        channel_type: the particular channel type to cluster by proximity.

    Returns:
        List[List[bipolar_cell]]: A list of clusters.
    """
    unassigned = bipolar_cells.copy()
    clusters = []

    while unassigned:
        # Take the first unassigned cell as seed
        seed = unassigned.pop(0)
        seed_pos = np.array([seed.x, seed.y])
        # Compute distances to all other unassigned cells
        dists = [np.linalg.norm(np.array([b.x, b.y]) - seed_pos) for b in unassigned]
        # Sort indices by distance
        sorted_idx = np.argsort(dists)
        # Take the nearest (group_size - 1) neighbors
        neighbors = [unassigned[i] for i in sorted_idx[:group_size - 1]]
        # Form cluster
        cluster = [seed] + neighbors
        # Remove chosen neighbors from unassigned
        for b in neighbors:
            unassigned.remove(b)
        clusters.append(cluster)

    return clusters


def initialize_midget_cells(cone_bipolar_cells, group_size=8, lambda_m=30.0, integration_factor=1.0, threshold=0.3):
    """
    Cluster r-g cone bipolar cells into groups and create Midget Ganglion Cells.
    """
    # Group midget bipolar cells.
    clusters = cluster_bipolar_by_proximity(cone_bipolar_cells, group_size)
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
    clusters = cluster_bipolar_by_proximity(cone_bipolar_cells, group_size)
    parasol_cells = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        pg = ParasolGanglion(cluster, lambda_p=lambda_p, integration_factor=integration_factor, threshold=threshold)
        pg.integrate()
        parasol_cells.append(pg)
    return parasol_cells


def initialize_small_bistratified_cells(cone_bipolar_cells, group_size=6, lambda_sb=60.0, integration_factor=1.0, threshold=0.3):
    """
    Cluster b-y cone bipolar cells into groups and create Small Bistratified Ganglion Cells.
    """
    # Group S bipolar cells
    clusters = cluster_bipolar_by_proximity(cone_bipolar_cells, group_size)
    small_bistratified_ganglion_cells = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        sbg = SmallBistratifiedGanglion(cluster, lambda_sb=lambda_sb,
                                        integration_factor=integration_factor, threshold=threshold)
        small_bistratified_ganglion_cells.append(sbg)
    return small_bistratified_ganglion_cells
