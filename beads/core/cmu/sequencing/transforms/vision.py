"""
The transform operation of the Push Implementation (refer Turing Beads.good-notes)
"""

import numpy as np


class Amacrine:
    def __init__(self, inhibition_factor=0.2, temporal_window=3):
        """
        Initialize an Amacrine cell.

        Args:
            inhibition_factor (float): Determines the strength of lateral inhibition.
            temporal_window (int): Number of iterations over which the cell integrates its input.
        """
        self.inhibition_factor = inhibition_factor
        self.temporal_window = temporal_window
        self.signal_history = []  # Stores recent bipolar inputs
        self.processed_stimulus = None  # Output after modulation and inhibition

    def modulate(self, bipolar_signal):
        """
        Integrate the bipolar cell signal over a short temporal window.
        This simulates the temporal integration of transient signals.

        Args:
            bipolar_signal (numpy.ndarray or float): Input signal from a bipolar cell.

        Returns:
            numpy.ndarray or float: Modulated signal.
        """
        # Append new signal and maintain the temporal window.
        self.signal_history.append(bipolar_signal)
        if len(self.signal_history) > self.temporal_window:
            self.signal_history.pop(0)

        # Compute a simple moving average over the window.
        modulated_signal = np.mean(self.signal_history, axis=0)
        return modulated_signal

    def inhibit(self, modulated_signal):
        """
        Apply additional lateral inhibition on the modulated signal.
        This simulates further sharpening of the contrast.

        Args:
            modulated_signal (numpy.ndarray or float): The modulated signal.

        Returns:
            numpy.ndarray or float: Inhibited signal.
        """
        # Inhibit the signal by subtracting a fraction determined by the inhibition factor.
        inhibited_signal = modulated_signal * (1.0 - self.inhibition_factor)
        # Clamp the result to non-negative values.
        inhibited_signal = np.maximum(inhibited_signal, 0)
        return inhibited_signal


def initialize_amacrine_cells(retina, temporal_window=3, inhibition_factor=0.2):
    """
    Creates amacrine cells by associating each bipolar cell with an amacrine cell.
    Each amacrine cell modulates and then inhibits the bipolar cell’s output.

    Args:
        retina (object): The retina object containing a bipolar cell layer (retina.bipolar_cells).
        temporal_window (int): Temporal integration window for modulation.
        inhibition_factor (float): Inhibition strength.

    Returns:
        retina: The retina object updated with an amacrine cell layer.
    """
    amacrine_cells = []
    for bipolar in retina.bipolar_cells:
        amacrine = Amacrine(inhibition_factor=inhibition_factor, temporal_window=temporal_window)
        # Process the bipolar cell's stimulus through modulation and inhibition.
        modulated = amacrine.modulate(bipolar.processed_stimulus)
        inhibited = amacrine.inhibit(modulated)
        amacrine.processed_stimulus = inhibited
        amacrine_cells.append(amacrine)

    retina.amacrine_cells = amacrine_cells
    return retina
