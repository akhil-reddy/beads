"""
The combine operation of the Push Implementation (refer Turing Beads.good-notes)

Before this operation, the RGB stimulus is converted into a lightweight "beehive" like structure. This structure
is analogous to a set of coloured rain drops separated by a membrane.

"""
import math

import numpy as np

"""
After conversion, each unit / "drop" is combined with its neighbours based on similarity. The membrane
is dissolved, and they form a larger "drop" i.e., the functionalities of horizontal cells are implemented here. 
Horizontal cells spread their signals laterally to inhibit nearby photoreceptors. The spread is dynamic and
influenced by neurotransmitters like dopamine. This “lateral inhibition” improves
contrast and sharpens edges. Essentially, before bipolar cell processing, horizontal cells makes the stimulus
interact constructively / destructively so that "context" spreads out horizontally.
"""


class Horizontal:
    def __init__(self, x, y, stimulus, area=1.0):
        self.x = x
        self.y = y
        self.stimulus = stimulus  # Expected to be a numpy array [R, G, B]
        self.field_stimulus = 0.0
        self.area = area
        self.pointers = []  # List of neighbouring Horizontal cells

    '''
    In a traditional ML sense, this operation is equivalent to channel preprocessing / convolution in a
    Convolutional Neural Network (CNN).
    
    Consider this datastructure if needed:
    
    {
        area: NUMBER, provides flexibility on the shape of the drop without describing the exact
            shape, which is of secondary significance
        *pointers: LIST of all neighbours, but carefully mapped to a "sub number" in the area.
            This esoteric structure should be equivalent to the Turing implementation of a "drop".
        stimulus: DATA, contains the RGB transformed structure for stimulus.
    }
    '''

    def link(self, horizontal_cells, radius=10.0):  # radius is in microns
        """
        Link this horizontal cell to neighbours within a given radius.

        Args:
            horizontal_cells (list): List of all Horizontal cells.
            radius (float): Maximum distance to consider a neighbour.
        """
        for cell in horizontal_cells:
            if cell is not self:
                distance = math.hypot(self.x - cell.x, self.y - cell.y)
                if distance <= radius:
                    self.pointers.append(cell)

    def center_calculate(self, k_n=1.0, lambda_n=175.0):  # lambda in microns
        """
        Calculate center stimulus from receptor field.

        Args:
            k_n (float): Amplitude for the narrow (center) component.
            lambda_n (float): Decay constant for the narrow component.

        Returns:
            numpy.ndarray: The receptor stimulus (ensured non-negative).
        """
        if not self.pointers:
            self.field_stimulus = self.stimulus

        # The receptor field includes the photoreceptor directly under the horizontal cell
        narrow_weighted_sum = self.stimulus * k_n
        narrow_total_weight = k_n

        # Compute the weight for each neighbor and accumulate weighted stimuli
        for cell in self.pointers:
            d = math.hypot(self.x - cell.x, self.y - cell.y)

            narrow_weight = k_n * math.exp(-d / lambda_n)
            narrow_weighted_sum += cell.stimulus * narrow_weight
            narrow_total_weight += narrow_weight

        narrow_weighted_avg = narrow_weighted_sum / narrow_total_weight
        self.field_stimulus = narrow_weighted_avg

    '''
    In a traditional ML sense, this operation is equivalent to the convolutional operator
    in a CNN. However, the analogy is made purely for understanding purposes. 
    '''

    def surround_inhibit(self, inhibition_strength=0.1, k_w=0.5, lambda_w=700.0):  # lambda in microns
        """
        Apply lateral inhibition from gap junction horizontal cells.

        Args:
            inhibition_strength (float): Neurotransmitter factor for the net neighbor influence.
            k_w (float): Amplitude for the wide (surround) component.
            lambda_w (float): Decay constant for the wide component.

        Returns:
            numpy.ndarray: The new inhibited stimulus (ensured non-negative).
        """
        if not self.pointers:
            return self.field_stimulus

        # The horizontal cells coupled with gap junctions
        wide_weighted_sum = np.zeros_like(self.stimulus, dtype=float)
        wide_total_weight = 0.0

        # Compute the weight for each neighbor and accumulate weighted stimuli
        for cell in self.pointers:
            d = math.hypot(self.x - cell.x, self.y - cell.y)

            wide_weight = k_w * math.exp(-d / lambda_w)
            wide_weighted_sum += cell.stimulus * wide_weight
            wide_total_weight += wide_weight

        wide_weighted_avg = wide_weighted_sum / wide_total_weight

        # Subtract a fraction of the weighted average from the current stimulus to apply inhibition.
        inhibited_stimulus = self.field_stimulus - inhibition_strength * wide_weighted_avg

        # Ensure the resulting stimulus remains non-negative.
        inhibited_stimulus = np.maximum(inhibited_stimulus, 0)
        return inhibited_stimulus


def initialize_horizontal_cells(retina, inhibition_radius=10.0):
    """
    Creates horizontal cells from the existing photoreceptors (the “drops”).
    Each photoreceptor cell is assigned a horizontal cell with a default stimulus.
    In a complete implementation, the stimulus would be derived from the RGB-transformed input.

    Args:
        retina (object): The retina object, which already has a list of photoreceptor cells.
        inhibition_radius (float): The radius within which horizontal cells are linked.

    Returns:
        retina: The retina object updated with horizontal cells.
    """
    horizontal_cells = []
    for cell in retina.photoreceptor_cells:
        if cell.cell_type == "cone":
            # In a full run, cell would already have a stimulus attribute.
            # Here we initialize with a default (e.g., zero stimulus for [R,G,B]).
            default_stimulus = np.array([0.0, 0.0, 0.0])
            horizontal_cells.append(Horizontal(cell.x, cell.y, default_stimulus))

    # Link each horizontal cell to its neighbours based on the inhibition_radius.
    for h_cell in horizontal_cells:
        h_cell.link(horizontal_cells, radius=inhibition_radius)

    # Attach the horizontal cell layer to the retina.
    retina.init_horizontal_cells(horizontal_cells)
    return retina


"""
Bipolar cells further refine the laterally inhibited stimulus through ON-OFF push-pull mechanism. Furthermore,
they prepare the graded stimulus for spike transmission by enhancing the stimulus at high details and inhibiting
stimulus when its at low detail. Neurotransmitters are highly active here. 
"""


class Bipolar:
    def __init__(self, x, y, cell_type='ON', threshold=0.5, gain=2.0, tau=0.1, saturation=1.0):
        """
        Initialize a bipolar cell that processes graded input over time.

        Args:
            cell_type (str): 'ON' or 'OFF'. ON bipolar cells depolarize in response to a reduction
                             of glutamate (i.e. when the photoreceptor input decreases), whereas OFF bipolar cells
                             depolarize when the glutamate input increases.
            threshold (float): The baseline threshold level.
            gain (float): The amplification factor for signals exceeding the threshold.
            tau (float): Time constant (in seconds) for the membrane's leaky integration, reflecting temporal filtering.
            saturation (float): Maximum possible output (models the limited dynamic range).
        """
        self.cell_type = cell_type.upper()
        self.x = x
        self.y = y
        self.threshold = threshold
        self.gain = gain
        self.tau = tau  # Time constant (s) for integration
        self.saturation = saturation
        self.V = 0.0  # Membrane potential (or output) at the current time
        self.output_history = []  # To store the temporal evolution of responses

    def update(self, input_signal, dt):
        """
        Update the bipolar cell's membrane potential given a graded input signal.
        This function simulates a leaky integrator (RC circuit) to reflect the temporal dynamics of bipolar cells.

        Args:
            input_signal (float or np.ndarray): The instantaneous graded input (normalized, e.g. 0 to 1) f
            rom photoreceptor/horizontal cell processing.
            dt (float): The time step (in seconds) for numerical integration.

        Returns:
            float: The updated (non-linear) output of the cell.
        """
        # For ON bipolar cells, the reduction in photoreceptor glutamate (i.e., a lower input)
        # produces excitation. For OFF bipolar cells, an increase in input is excitatory.
        if self.cell_type == 'ON':
            # Compute the drive as the difference between threshold and input
            drive = self.threshold - input_signal
        elif self.cell_type == 'OFF':
            drive = input_signal - self.threshold
        else:
            raise ValueError("cell_type must be either 'ON' or 'OFF'.")

        # Only positive differences (excitatory drive) contribute:
        drive = max(drive, 0)
        # Non-linear amplification:
        drive *= self.gain

        # Update the membrane potential using a leaky integrator:
        # dV/dt = (-V + drive) / tau
        dV = (-self.V + drive) * (dt / self.tau)
        self.V += dV

        # Enforce saturation limits:
        self.V = np.clip(self.V, 0, self.saturation)

        self.output_history.append(self.V)
        return self.V

    def get_output(self):
        """Return the current output (membrane potential) of the bipolar cell."""
        return self.V


def initialize_rod_bipolar_cells(retina):
    """
    Given a retina object that contains a list of rod photoreceptor cells, create an ON bipolar cell layer.

    Args:
        retina: An object that has an attribute `photoreceptor_cells`, a list of photoreceptor cell objects.

    Returns:
        retina: The retina object updated with rod bipolar cells stored in `retina.rod_bipolar_cells`.
    """
    bipolar_cells = []

    for cell in retina.photoreceptor_cells:
        if cell.cell_type == "rod":
            # Create an ON bipolar cell with typical parameter values
            bipolar = Bipolar(cell.x, cell.y, cell_type='ON', threshold=0.5, tau=0.07, gain=3.0, saturation=1.0)
            bipolar_cells.append(bipolar)

    retina.rod_bipolar_cells = bipolar_cells
    return retina


def initialize_cone_bipolar_cells(retina):
    """
    Given a retina object that contains a list of horizontal cells with already computed inhibitory responses,
    create a bipolar cell layer.

    Args:
        retina: An object that has an attribute `horizontal_cells`, a list of horizontal cell objects,
                each with a computed attribute `inhibited_stimulus` (range normalized to [0, 1]).

    Returns:
        retina: The retina object updated with cone bipolar cells stored in `retina.cone_bipolar_cells`.
    """
    bipolar_cells = []

    for h_cell in retina.horizontal_cells:
        # Create an ON bipolar cell with typical parameter values
        bipolar = Bipolar(h_cell.x, h_cell.y, cell_type='ON', threshold=0.5, tau=0.07, gain=3.0, saturation=1.0)
        bipolar_cells.append(bipolar)

        # Create an OFF bipolar cell with typical parameter values
        bipolar = Bipolar(h_cell.x, h_cell.y, cell_type='OFF', threshold=0.5, tau=0.07, gain=3.0, saturation=1.0)
        bipolar_cells.append(bipolar)

    retina.cone_bipolar_cells = bipolar_cells
    return retina
