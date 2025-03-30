import numpy as np

"""
4 types of Ganglion Cells:
1. Midget Cells - Small receptive fields that support high-acuity vision and color opponency (red–green)
2. Parasol Cells - High sensitivity to motion and contrast, with rapid responses
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
        return net_vector

    def update_bipolar_firing(self, max_proj=6.0):  # max_proj is DSGC focus radius in microns
        """
        For each SAC in the DSGC's cluster, iterate over its bipolar cells.
        For each bipolar cell, compute the projection of its relative position (to DSGC center)
        onto the DSGC's integrated 2D vector. If this projection is high, increase the bipolar cell's
        firing rate (for demonstration, add a proportional increment).

        This method simulates feedback where the DSGC modulates the firing rate of bipolar cells
        based on alignment with the integrated directional signal.
        """
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

    def spike(self):
        """
        Generate a spike based on the integrated directional signal.
        Uses a simple threshold: if the integrated signal exceeds the threshold, generate a spike.

        Returns:
            int: 1 if spike generated, 0 otherwise.
        """
        spike_output = 1 if self.integrated_signal > self.threshold else 0
        self.spikes.append(spike_output)
        return spike_output


###############################################################################
# DSGC Layer Initialization
###############################################################################

def initialize_DSGCs(retina, sac_group_size=5, lambda_ds=100.0, integration_factor=1.0, threshold=0.5):
    """
    Cluster SACs from the retina into groups of 'sac_group_size' and assign each group to one DSGC.

    Args:
        retina (object): Retina object with retina.starburst_amacrine_cells.
        sac_group_size (int): Number of SACs per DSGC.
        lambda_ds (float): Spatial decay constant (micrometers) for DSGC integration.
        integration_factor (float): Scaling factor for integration.
        threshold (float): DSGC spiking threshold.

    Returns:
        retina: Updated retina object with retina.ganglion_cells set.
    """
    # Cluster SACs (simple grouping by order).
    sac_clusters = []
    current_cluster = []
    for sac in retina.starburst_amacrine_cells:
        current_cluster.append(sac)
        if len(current_cluster) >= sac_group_size:
            sac_clusters.append(current_cluster)
            current_cluster = []
    if current_cluster:
        sac_clusters.append(current_cluster)

    dsgc_list = []
    for cluster in sac_clusters:
        dsgc = DSGanglion(sac_cluster=cluster, lambda_ds=lambda_ds,
                          integration_factor=integration_factor, threshold=threshold)
        dsgc.integrate_SACs()
        dsgc_list.append(dsgc)

    retina.ganglion_cells = dsgc_list
    return retina
