###############################################################################
# Ribbon Synapse
###############################################################################
import numpy as np


class RibbonSynapse:
    def __init__(self, release_threshold=-50.0, nonlin_gain=0.2):
        """
        Models the ribbon synapse of the inner hair cell.
        Converts the graded receptor potential into a rate of neurotransmitter release.

        Args:
            release_threshold (float): The membrane potential (mV) above which transmitter release increases.
            nonlin_gain (float): Gain for the nonlinear (sigmoidal) relationship.

        Research Basis:
            - Moser et al. (2006); Glowatzki & Fuchs (2002): Discuss ribbon synapse dynamics.
        """
        self.release_threshold = release_threshold
        self.nonlin_gain = nonlin_gain

    def release_rate(self, ihc_potential):
        """
        Compute neurotransmitter release rate as a function of the IHC receptor potential.

        A sigmoidal function is used to model the nonlinear release properties.

        Args:
            ihc_potential (float): Receptor potential from an inner hair cell (mV).

        Returns:
            float: Neurotransmitter release rate (arbitrary units, e.g., vesicles/s).
        """
        # Sigmoidal relationship: more release as potential exceeds threshold.
        rate = 1.0 / (1.0 + np.exp(-self.nonlin_gain * (ihc_potential - self.release_threshold)))
        return rate
