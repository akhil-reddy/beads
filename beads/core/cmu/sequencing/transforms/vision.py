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


###############################################################################
# Starburst Amacrine Cell Model (Directional)
###############################################################################

"""
This needs an overhaul. Starburst class should have focus area and dendrites in all directions.
Direction is calculated from leaky spatial centrifugal integrator.
If many directions align, DSGC should increase the rate of firing appropriately.
"""


class StarburstAmacrine(GenericAmacrine):
    def __init__(self, preferred_direction, directional_sigma,
                 tau=0.05, V_rest=-65.0, V_threshold=-50.0,
                 gain=1.2, g_max=1.0, sigmoid_slope=0.3):
        """
        Starburst amacrine cell model incorporating directional asymmetry.

        In addition to the standard dynamics, this cell weights bipolar inputs based on their spatial
        positions relative to a preferred direction (in radians). This models the asymmetry in the dendritic
        arbor of starburst amacrine cells which underlies directional selectivity.

        Parameters:
            preferred_direction (float): Preferred direction (radians) of input integration.
            directional_sigma (float): Standard deviation (radians) of the directional weighting function.
            (Other parameters are as in GenericAmacrineCell.)
        """
        super().__init__(tau=tau, V_rest=V_rest, V_threshold=V_threshold,
                         gain=gain, g_max=g_max, sigmoid_slope=sigmoid_slope)
        self.preferred_direction = preferred_direction  # in radians
        self.directional_sigma = directional_sigma

    def directional_weight(self, bipolar_position, cell_position):
        """
        Compute a directional weight based on the angle between the vector from the amacrine cell
        to the bipolar cell and the cell's preferred direction.

        Args:
            bipolar_position (tuple): (x, y) position of the bipolar input.
            cell_position (tuple): (x, y) position of the amacrine cell.

        Returns:
            float: Weight between 0 and 1, with 1 for inputs aligned with the preferred direction.
        """
        # Compute the vector from the cell to the bipolar input.
        dx = bipolar_position[0] - cell_position[0]
        dy = bipolar_position[1] - cell_position[1]
        # Compute the angle of this vector.
        angle = np.arctan2(dy, dx)
        # Compute the angular difference from the preferred direction.
        d_angle = np.angle(np.exp(1j * (angle - self.preferred_direction)))
        # Use a Gaussian function of the angular difference.
        weight = np.exp(-0.5 * (d_angle / self.directional_sigma) ** 2)
        return weight

    def update_directional(self, bipolar_inputs, bipolar_positions, cell_position, dt):
        """
        Update the membrane potential by integrating bipolar inputs with directional weighting.

        Parameters:
            bipolar_inputs (list of float): Graded inputs from bipolar cells (normalized 0–1).
            bipolar_positions (list of tuple): Positions (x,y) for each bipolar input.
            cell_position (tuple): Position (x,y) of this amacrine cell.
            dt (float): Time step (s).

        Returns:
            float: Inhibitory output computed as before.
        """
        # Compute weights for each bipolar input based on their position.
        weights = [self.directional_weight(pos, cell_position) for pos in bipolar_positions]
        # Compute the effective input as the weighted average.
        if np.sum(weights) > 0:
            effective_input = np.dot(bipolar_inputs, weights) / np.sum(weights)
        else:
            effective_input = 0.0

        # Update membrane potential with the effective (weighted) input.
        return self.update(effective_input, dt)


def initialize_starburst_amacrine_cells(retina, **params):
    """
    Create an amacrine cell layer associated with each bipolar cell in the retina.

    Args:
        retina (object): Should contain a list retina.cone_bipolar_cells.
        params: Parameters for the chosen amacrine cell model.

    Returns:
        retina: The retina object with retina.aii_amacrine_cells set.
    """
    amacrine_cells = []
    # For simplicity, assume that each bipolar cell gives rise to one amacrine cell.
    # In the starburst case, we need spatial info. Here we assume each bipolar cell has attributes
    # processed_stimulus and position.
    for bipolar in retina.cone_bipolar_cells:
        # For starburst cells, we require additional spatial info.
        if not hasattr(bipolar, 'position'):
            raise ValueError("Bipolar cell must have a 'position' attribute for starburst model.")
        # TODO: Overhaul
        cell = StarburstAmacrine(preferred_direction=0.0, directional_sigma=0.5, **params)
        cell.update_directional([bipolar.processed_stimulus],
                                [bipolar.position],
                                bipolar.position,
                                dt=0.01)
        amacrine_cells.append(cell)

    retina.starburst_amacrine_cells = amacrine_cells
    return retina
