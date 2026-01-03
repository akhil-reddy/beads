"""
The combine operation of the Push Implementation (refer Turing Beads.good-notes)

Before this operation, the RGB stimulus is converted into a lightweight "beehive" like structure. This structure
is analogous to a set of coloured rain drops separated by a membrane.

"""
import argparse
import math
import pickle
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from beads.core.cmu.sequencing.receive.vision import Cell

"""
After conversion, each unit / "drop" is combined with its neighbours based on similarity. The membrane
is dissolved, and they form a larger "drop" i.e., the functionalities of horizontal cells are implemented here. 
Horizontal cells spread their signals laterally to inhibit nearby photoreceptors. The spread is dynamic and
influenced by neurotransmitters like dopamine. This “lateral inhibition” improves
contrast and sharpens edges. Essentially, before bipolar cell processing, horizontal cells makes the stimulus
interact constructively / destructively so that "context" spreads out horizontally.
"""


class Horizontal:
    def __init__(self, x, y, subtype, photoreceptor_cell=None, pointers=None, latest=None):
        self.x = x
        self.y = y
        self.subtype = subtype
        self.photoreceptor_cell = photoreceptor_cell
        self.stimulus = 0.0
        self.field_stimulus = 0.0
        if pointers is not None:
            self.pointers = pointers
        else:
            self.pointers = []  # List of neighbouring Horizontal cells
        self.latest = latest

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

    def link(self, neighbors):  # radius is in microns
        """
        Link this horizontal cell to neighbors within a given radius.

        Args:
            neighbors (list): List of all neighboring cells including self.
        """
        # Remove reference to self from the list of pointers.
        self.pointers = neighbors[1:]

    def set_stimulus(self, stimulus):  # lambda in microns
        """
        Set center stimulus

        Args:
            stimulus (float): Stimulus value.
        """
        self.stimulus = stimulus

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
            return self.stimulus

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
        self.field_stimulus = wide_weighted_avg

        # Subtract a fraction of the weighted average from the current stimulus to apply inhibition.
        inhibited_stimulus = self.stimulus - inhibition_strength * wide_weighted_avg

        # Ensure the resulting stimulus remains non-negative.
        inhibited_stimulus = np.maximum(inhibited_stimulus, 0)
        self.latest = inhibited_stimulus
        return self.latest

    def function(self, stimulus):
        self.set_stimulus(stimulus)
        return self.surround_inhibit()


def initialize_horizontal_cells(photoreceptor_cells, inhibition_radius=10.0):
    """
    Creates horizontal cells from the existing photoreceptors (the “drops”).
    Each photoreceptor cell is assigned a horizontal cell with a default stimulus.
    In a complete implementation, the stimulus would be derived from the RGB-transformed input.

    Args:
        photoreceptor_cells (list): A list of photoreceptor cells.
        inhibition_radius (float): The radius within which horizontal cells are linked.

    Returns:
        cells: The horizontal cells
    """
    horizontal_cells = []
    positions = []
    for cell in photoreceptor_cells:
        if cell.cell_type == "cone":
            for cone_cell in cell.cells:
                horizontal_cells.append(Horizontal(cone_cell.x, cone_cell.y, cone_cell.subtype, photoreceptor_cell=cone_cell))
                positions.append([float(cell.x), float(cell.y)])

    positions = np.asarray(positions, dtype=np.float32)
    tree = KDTree(positions)
    neighbors_idxs = tree.query_ball_point(positions, r=inhibition_radius, workers=-1)

    # Link each horizontal to its neighbours (exclude self index)
    for idx, neigh_idxs in enumerate(neighbors_idxs):
        neighs = [horizontal_cells[i] for i in neigh_idxs if i != idx]
        horizontal_cells[idx].link(neighs)

    # Attach the horizontal cell layer to the retina.
    return horizontal_cells


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

    def update(self, input_signal, dt=0.1):
        """
        Update the bipolar cell's membrane potential given a graded input signal.
        This function simulates a leaky integrator (RC circuit) to reflect the temporal dynamics of bipolar cells.

        Args:
            input_signal (float or np.ndarray): The instantaneous graded input (normalized, e.g. 0 to 1) f
            rom photoreceptor/horizontal cell processing.
            dt (float): The time step (in seconds) for numerical integration.
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

    def get_output(self):
        """Return the current output (membrane potential) of the bipolar cell."""
        return self.V

    def function(self, input_stimulus):
        self.update(input_stimulus)
        return self.get_output()


def initialize_rod_bipolar_cells(photoreceptor_cells):
    """
    Given a retina object that contains a list of rod photoreceptor cells, create an ON bipolar cell layer.

    Args:
        photoreceptor_cells: A list of photoreceptor cell objects.

    Returns:
        rod_bipolar_cells: The rod bipolar cells
    """
    rod_bipolar_cells = []

    for cell in photoreceptor_cells:
        if cell.cell_type == "rod":
            # Create an ON bipolar cell with typical parameter values
            bipolar = Bipolar(cell.x, cell.y, cell_type='ON', threshold=0.5, tau=0.07, gain=3.0, saturation=1.0)
            rod_bipolar_cells.append(bipolar)

    return rod_bipolar_cells


def serialize_horizontal_cells(horizontal_cells: List[object], out_path: str):
    """
    Serialize a non-recursive representation:
      - For each Horizontal: store (x, y, stimulus, neighbors as indices).
    This builds a map id->index and stores lists of integers instead of object refs.
    """
    # build index map for quick lookup
    idx_map = {id(h): i for i, h in enumerate(horizontal_cells)}

    serial = []
    for i, h in enumerate(horizontal_cells):
        # gather neighbor indices (skip neighbors not in this list)
        neigh_idxs = []
        for n in getattr(h, 'pointers', []):
            if id(n) in idx_map:
                neigh_idxs.append(idx_map[id(n)])
        serial.append({
            'x': float(getattr(h, 'x', 0.0)),
            'y': float(getattr(h, 'y', 0.0)),
            'subtype': getattr(h, 'subtype', ''),
            'latest': getattr(h, 'latest', 0.0),
            'neighbors': neigh_idxs,
            'photoreceptor_cell': getattr(h, 'photoreceptor_cell', None),
        })

    with open(out_path, 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(serial, file, protocol=pickle.HIGHEST_PROTOCOL)


def test():
    p = argparse.ArgumentParser()
    p.add_argument("--out_csv", default="/Users/akhilreddy/IdeaProjects/beads/out/visual/horizontal_out.csv")
    args = p.parse_args()

    with open('/Users/akhilreddy/IdeaProjects/beads/out/visual/photoreceptors.pkl', 'rb') as file:
        photoreceptor_cells = pickle.load(file)

    """
    
    rod_bipolar_cells = initialize_rod_bipolar_cells(photoreceptor_cells)
    
    records = []
    idx = 0
    for r, p in zip(rod_bipolar_cells, photoreceptor_cells):
        response = r.function(p.cell.latest)
        records.append({
            "idx": idx,
            "x_micron": float(r.x),
            "y_micron": float(r.y),
            "response": response
        })
        idx += 1
    
    with open('/Users/akhilreddy/IdeaProjects/beads/out/visual/rod_bipolar.pkl', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(rod_bipolar_cells, file)

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote CSV: {args.out_csv}  (n_cells = {len(df)})")
        
    """

    horizontal_cells = initialize_horizontal_cells(photoreceptor_cells)

    records = []
    for idx, c in enumerate(horizontal_cells):
        c.set_stimulus(c.photoreceptor_cell.cell.latest)

    for idx, c in enumerate(horizontal_cells):
        before = c.stimulus
        c.surround_inhibit()
        records.append({
            "idx": idx,
            "x_micron": float(c.x),
            "y_micron": float(c.y),
            "subtype": c.photoreceptor_cell.subtype,
            "stimulus_before": before,
            "field_stimulus": c.field_stimulus,
            "stimulus_after": c.latest
        })

    serialize_horizontal_cells(horizontal_cells, '/Users/akhilreddy/IdeaProjects/beads/out/visual/horizontal.pkl')

    df = pd.DataFrame.from_records(records)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote CSV: {args.out_csv}  (n_cells = {len(df)})")


if __name__ == "__main__":
    test()
    # pass
