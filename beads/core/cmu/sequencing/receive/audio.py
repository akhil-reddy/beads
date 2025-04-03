###############################################################################
# Inner Hair Cell (IHC)
###############################################################################
class InnerHairCell:
    def __init__(self, resting_potential=-70.0, max_potential=0.0, sensitivity=0.05):
        """
        Simulates an inner hair cell that converts mechanical vibrations (from the basilar membrane)
        into a graded electrical receptor potential.

        Args:
            resting_potential (float): The baseline membrane potential (in mV).
            max_potential (float): The maximum receptor potential achievable (in mV).
            sensitivity (float): Sensitivity factor relating mechanical stimulus amplitude
                                 (in arbitrary units) to voltage change.

        Research Basis:
            - Hudspeth (1989), Dallos (1992): Describe the receptor potential properties
              of inner hair cells.
        """
        self.resting_potential = resting_potential
        self.max_potential = max_potential
        self.sensitivity = sensitivity
        self.potential = resting_potential

    def transduce(self, mechanical_stimulus):
        """
        Transduce a mechanical stimulus into a receptor potential.

        A saturating nonlinearity is applied to simulate the limited range of receptor potentials.

        Args:
            mechanical_stimulus (float): Amplitude of mechanical stimulation.

        Returns:
            float: The receptor potential (mV).
        """
        # Linear conversion with saturation.
        delta = self.sensitivity * mechanical_stimulus
        self.potential = self.resting_potential + delta
        # Ensure the potential does not exceed max_potential.
        self.potential = min(self.potential, self.max_potential)
        return self.potential


###############################################################################
# Outer Hair Cell (OHC)
###############################################################################
class OuterHairCell:
    def __init__(self, amplification_factor=2.0):
        """
        Simulates an outer hair cell that provides active amplification
        through electromotility.

        Args:
            amplification_factor (float): Factor by which the OHC amplifies mechanical input.

        Research Basis:
            - Dallos (1992); Ashmore (2008): Describe the electromotile properties of outer hair cells.
        """
        self.amplification_factor = amplification_factor
        self.length_change = 0.0

    def amplify(self, mechanical_stimulus):
        """
        Amplify the mechanical stimulus via electromotility.

        Args:
            mechanical_stimulus (float): Input mechanical stimulus amplitude.

        Returns:
            float: The amplified stimulus.
        """
        amplified = self.amplification_factor * mechanical_stimulus
        # For simplicity, store the change in length.
        self.length_change = amplified
        return amplified
