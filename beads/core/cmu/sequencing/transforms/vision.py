import numpy as np


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
    def __init__(self, tau=0.04, V_rest=-65.0, V_threshold=-52.0,
                 gain=1.2, g_gap_max=1.0, g_inhib_max=0.8, sigmoid_slope=0.4):
        """
        AII amacrine cell model, specialized for scotopic rod signal processing.

        AII cells receive input exclusively from rod bipolar cells and then:
          - Electrically couple to ON cone bipolar cells (linear transfer).
          - Inhibit OFF cone bipolar cells via glycinergic synapses (nonlinear, steep threshold).

        Parameters:
            tau (float): Membrane time constant (s) (typically ~40 ms).
            V_rest (float): Resting membrane potential (mV).
            V_threshold (float): Threshold potential (mV) for glycinergic release.
            gain (float): Gain factor converting rod bipolar input.
            g_gap_max (float): Maximum output for electrical coupling.
            g_inhib_max (float): Maximum inhibitory output for glycinergic transmission.
            sigmoid_slope (float): Slope of the sigmoidal nonlinearity.
        """
        self.tau = tau
        self.V_rest = V_rest
        self.V_threshold = V_threshold
        self.gain = gain
        self.g_gap_max = g_gap_max
        self.g_inhib_max = g_inhib_max
        self.sigmoid_slope = sigmoid_slope

        self.V = V_rest
        self.V_history = []

    def update(self, rod_bipolar_input, dt):
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

        return electrical_output, inhibitory_output

    def reset(self):
        self.V = self.V_rest
        self.V_history = []


def initialize_aii_amacrine_cells(retina, **params):
    """
    Create an amacrine cell layer associated with each bipolar cell in the retina.

    Args:
        retina (object): Should contain a list retina.rod_bipolar_cells.
        params: Parameters for the chosen amacrine cell model.

    Returns:
        retina: The retina object with retina.aii_amacrine_cells set.
    """
    amacrine_cells = []
    for bipolar in retina.rod_bipolar_cells:
        cell = AIIAmacrine(**params)
        cell.update(bipolar.processed_stimulus, dt=0.01)
        amacrine_cells.append(cell)

    retina.aii_amacrine_cells = amacrine_cells
    return retina


def exponential_decay(distance, lambda_r):
    """Compute an exponential decay weight based on distance."""
    return np.exp(-distance / lambda_r)


def unit_vector(angle):
    """Return a 2D unit vector for a given angle (radians)."""
    return np.array([np.cos(angle), np.sin(angle)])


def cluster_bipolar_cells(bipolar_cells, distance_threshold=50.0, min_cluster_size=0):  # distance is in microns
    """
    Group bipolar cells into clusters based on their (x, y) positions.

    This simple greedy algorithm iterates through the list of bipolar cells and
    assigns each cell to an existing cluster if its distance from the cluster's centroid
    is below the threshold. Otherwise, it starts a new cluster.

    Args:
        bipolar_cells (list): List of bipolar cell objects (each must have .x and .y attributes).
        distance_threshold (float): Maximum distance (in same units as x,y) for grouping.
        min_cluster_size (int): Minimum number of cells for a cluster to be used.

    Returns:
        list: A list of clusters, where each cluster is a list of bipolar cells.
    """
    clusters = []

    for cell in bipolar_cells:
        if cell.cell_type == 'OFF':  # Avoid OFF cells for directional weighting
            continue
        cell_pos = np.array([cell.x, cell.y])
        added = False
        # Try to add the cell to an existing cluster.
        for cluster in clusters:
            # Compute cluster centroid.
            positions = np.array([[b.x, b.y] for b in cluster])
            centroid = np.mean(positions, axis=0)
            distance = np.linalg.norm(cell_pos - centroid)
            if distance <= distance_threshold:
                cluster.append(cell)
                added = True
                break
        if not added:
            clusters.append([cell])

    # Optionally, discard clusters smaller than min_cluster_size.
    valid_clusters = [cluster for cluster in clusters if len(cluster) >= min_cluster_size]
    return valid_clusters


###############################################################################
# Starburst Amacrine Cell Model (Directional, with radial grouping)
###############################################################################

class StarburstAmacrine:
    def __init__(self, bipolar_cells, lambda_r=50.0, nonlin_gain=10.0, nonlin_thresh=0.2, tau=0.05):
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
        self.preferred_direction = None
        self.centrifugal_output = 0.0

    def compute_directional_contributions(self):
        """
        Compute the individual weighted contribution vectors from each bipolar cell.

        For each bipolar cell:
          - Compute the relative vector from the SAC center.
          - Weight the bipolar output by an exponential decay based on distance.
          - Compute the contribution vector (weighted unit vector).

        Returns:
            numpy.ndarray: A 4D tensor of shape (1, N, 1, 2), where N is the number of bipolar cells.
        """
        contributions = []
        for b in self.bipolar_cells:
            bipolar_pos = np.array([b.x, b.y])
            r_vec = bipolar_pos - self.center
            r = np.linalg.norm(r_vec)
            # If r is zero (unlikely), skip the cell.
            if r == 0:
                contributions.append(np.array([0.0, 0.0]))
                continue
            # Weight = bipolar signal * exponential decay with distance.
            weight = b.processed_stimulus * exponential_decay(r, self.lambda_r)
            # Contribution vector = weight * (unit vector in direction of r_vec)
            contribution = weight * (r_vec / r)
            contributions.append(contribution)

        # Convert to numpy array; shape will be (N, 2)
        contributions = np.array(contributions)
        # Expand dimensions to make a 4D array: (1, N, 1, 2)
        self.contribution_tensor = contributions[np.newaxis, :, np.newaxis, :]
        return self.contribution_tensor


###############################################################################
# Initialization Function for Starburst Amacrine Layer (with Radial Grouping)
###############################################################################

def initialize_starburst_amacrine_cells(retina, distance_threshold=50.0, min_cluster_size=8,
                                        lambda_r=50.0, nonlin_gain=10.0, nonlin_thresh=0.2, tau=0.05):
    """
    Group cone bipolar cells based on their (x,y) coordinates into clusters (focus areas).
    Create a starburst amacrine cell for each cluster.

    Args:
        retina (object): Retina object with retina.cone_bipolar_cells (each with attributes: x, y, processed_stimulus).
        distance_threshold (float): Maximum distance for bipolar cells to be grouped into the same SAC.
        min_cluster_size (int): Minimum number of bipolar cells required for a group.
        lambda_r: Parameters for the SAC model.
        nonlin_gain: Parameters for the SAC model.
        nonlin_thresh: Parameters for the SAC model.
        tau: Parameters for the SAC model.

    Returns:
        retina: Updated retina object with retina.starburst_amacrine_cells.
    """
    clusters = cluster_bipolar_cells(retina.cone_bipolar_cells, distance_threshold, min_cluster_size)
    starburst_cells = []
    for cluster in clusters:
        sac = StarburstAmacrine(cluster, lambda_r=lambda_r, nonlin_gain=nonlin_gain,
                                nonlin_thresh=nonlin_thresh, tau=tau)
        starburst_cells.append(sac)
    retina.starburst_amacrine_cells = starburst_cells
    return retina
