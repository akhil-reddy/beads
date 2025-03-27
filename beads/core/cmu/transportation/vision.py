import numpy as np


class DSGanglion:
    def __init__(self, threshold=0.5, integration_factor=1.0):
        """
        Initialize a Direction Selective Ganglion cell.

        Args:
            threshold (float): Minimum integrated signal required to trigger a spike.
            integration_factor (float): Factor to scale the input before thresholding.
        """
        self.threshold = threshold
        self.integration_factor = integration_factor
        self.integrated_signal = None
        self.spikes = []  # History of spike outputs

    def integrate(self, amacrine_signal):
        """
        Integrate the input signal from an amacrine cell.
        This can be viewed as spatial or temporal integration before spiking.

        Args:
            amacrine_signal (numpy.ndarray or float): Input from an amacrine cell.

        Returns:
            numpy.ndarray or float: The integrated signal.
        """
        # Apply a simple scaling for integration.
        self.integrated_signal = self.integration_factor * amacrine_signal
        return self.integrated_signal

    def spike(self, integrated_signal):
        """
        Generate a spike based on the integrated signal.
        Here we use a simple threshold function.

        Args:
            integrated_signal (numpy.ndarray or float): The integrated input signal.

        Returns:
            int: 1 if a spike is generated, 0 otherwise.
        """
        # For simplicity, treat the integrated signal as a scalar by taking its mean if needed.
        value = np.mean(integrated_signal) if isinstance(integrated_signal, np.ndarray) else integrated_signal
        output_spike = 1 if value > self.threshold else 0
        self.spikes.append(output_spike)
        return output_spike


def initialize_ganglion_cells(retina, threshold=0.5, integration_factor=1.0):
    """
    Creates ganglion cells based on the amacrine cell layer.
    Each ganglion cell integrates the processed signal from its corresponding amacrine cell
    and produces a spiking output.

    Args:
        retina (object): The retina object containing an amacrine cell layer (retina.amacrine_cells).
        threshold (float): Spike generation threshold.
        integration_factor (float): Scaling factor for integration.

    Returns:
        retina: The retina object updated with a ganglion cell layer.
    """
    ganglion_cells = []
    for amacrine in retina.amacrine_cells:
        ganglion = DSGanglion(threshold=threshold, integration_factor=integration_factor)
        integrated = ganglion.integrate(amacrine.processed_stimulus)
        ganglion.spike(integrated)
        ganglion_cells.append(ganglion)

    retina.ganglion_cells = ganglion_cells
    return retina
