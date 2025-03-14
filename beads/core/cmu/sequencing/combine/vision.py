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

    def link(self, horizontal_cells, radius=10.0):
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

    '''
    In a traditional ML sense, this operation is equivalent to the convolutional operator
    in a CNN. However, the analogy is made purely for understanding purposes. 
    '''

    def inhibit(self, inhibition_strength=0.1):
        """
        Apply lateral inhibition by subtracting a fraction of the average neighbour stimulus.
        This is analogous to a convolution operation in a CNN.

        Args:
            inhibition_strength (float): Factor determining the strength of lateral inhibition.

        Returns:
            numpy.ndarray: The new, inhibited stimulus.
        """
        if not self.pointers:
            return self.stimulus

        # Compute the average stimulus from neighbours.
        total = np.zeros_like(self.stimulus)
        for cell in self.pointers:
            total += cell.stimulus
        avg_neighbour_stimulus = total / len(self.pointers)

        # Subtract a fraction of the average neighbour stimulus.
        inhibited_stimulus = self.stimulus - inhibition_strength * avg_neighbour_stimulus

        # Ensure the inhibited stimulus does not fall below zero.
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
    # Assume retina.cells is the list of photoreceptors.
    for cell in retina.cells:
        # In a full simulation, cell would already have a stimulus attribute.
        # Here we initialize with a default (e.g., zero stimulus for [R,G,B]).
        default_stimulus = np.array([0.0, 0.0, 0.0])
        horizontal_cells.append(Horizontal(cell.x, cell.y, default_stimulus))

    # Link each horizontal cell to its neighbours based on the inhibition_radius.
    for h_cell in horizontal_cells:
        h_cell.link(horizontal_cells, radius=inhibition_radius)

    # Attach the horizontal cell layer to the retina.
    retina.horizontal_cells = horizontal_cells
    return retina


"""
Bipolar cells further refine the laterally inhibited stimulus through ON-OFF push-pull mechanism. Furthermore,
they prepare the graded stimulus for spike transmission by enhancing the stimulus at high details and inhibiting
stimulus when its at low detail. Neurotransmitters are highly active here. 
"""


class Bipolar:
    def __init__(self, cell_type='ON', amplification_factor=2.0, inhibition_factor=0.5, detail_threshold=0.2):
        """
        Initialize a Bipolar cell that refines the laterally inhibited stimulus.

        Args:
            cell_type (str): 'ON' or 'OFF' type. ON cells are activated by increases in light intensity,
                             while OFF cells respond to decreases.
            amplification_factor (float): Factor used for non-linear amplification.
            inhibition_factor (float): Factor used for non-linear inhibition.
            detail_threshold (float): Threshold to differentiate high detail from low detail.
        """
        self.cell_type = cell_type
        self.amplification_factor = amplification_factor
        self.inhibition_factor = inhibition_factor
        self.detail_threshold = detail_threshold
        self.processed_stimulus = None  # Stores the refined stimulus for spike transmission

    def amplify(self, stimulus):
        """
        Applies non-linear amplification to the stimulus. Regions where the stimulus exceeds
        the detail_threshold are boosted to enhance high detail.

        Args:
            stimulus (numpy.ndarray or float): The input stimulus (e.g., an RGB vector or luminance value).

        Returns:
            numpy.ndarray or float: Amplified stimulus.
        """
        stim = np.array(stimulus, dtype=float)
        # For values above the threshold, boost using a power law.
        amplified = np.where(stim > self.detail_threshold,
                             np.power(stim, self.amplification_factor),
                             stim)
        return amplified

    def inhibit(self, stimulus):
        """
        Applies non-linear inhibition to the stimulus. Regions where the stimulus is below the
        detail_threshold are suppressed, simulating push-pull (ON-OFF) processing.

        Args:
            stimulus (numpy.ndarray or float): The input stimulus.

        Returns:
            numpy.ndarray or float: Inhibited stimulus.
        """
        stim = np.array(stimulus, dtype=float)
        # For values below the threshold, reduce the stimulus.
        inhibited = np.where(stim < self.detail_threshold,
                             stim * self.inhibition_factor,
                             stim)
        return inhibited


def initialize_bipolar_cells(retina, bipolar_inhibition_radius=5.0):
    """
    Creates bipolar cells based on the horizontal layer's laterally inhibited stimulus.
    In this simplified implementation, each horizontal cell is assigned a bipolar cell.
    The bipolar cell further refines the stimulus through non-linear amplification and inhibition.

    Args:
        retina (object): The retina object that holds, among other layers, the horizontal_cells.
        bipolar_inhibition_radius (float): Radius for potential integration (if combining multiple inputs).
                                          (Not used in this simple per-cell assignment but provided for extension.)

    Returns:
        retina: The retina object updated with a bipolar cell layer.
    """
    bipolar_cells = []
    # For each horizontal cell (each “drop”), create a corresponding bipolar cell.
    for h_cell in retina.horizontal_cells:
        # Retrieve the inhibited stimulus from the horizontal cell.
        # (Assumes h_cell.inhibit() returns an RGB vector or luminance value.)
        inhibited_stimulus = h_cell.inhibit()

        # For demonstration, choose bipolar cell type based on average stimulus intensity.
        # (This is an arbitrary criterion; in practice, ON and OFF cells are determined by circuit wiring.)
        avg_intensity = np.mean(inhibited_stimulus)
        cell_type = 'ON' if avg_intensity >= 0.5 else 'OFF'

        bipolar = Bipolar(cell_type=cell_type)
        # Sequentially process the stimulus: first amplify, then apply inhibition.
        amplified = bipolar.amplify(inhibited_stimulus)
        processed = bipolar.inhibit(amplified)
        bipolar.processed_stimulus = processed
        bipolar_cells.append(bipolar)

    retina.bipolar_cells = bipolar_cells
    return retina
