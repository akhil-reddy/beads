###############################################################################
# Neural Transducer
###############################################################################
class NeuralTransducer:
    def __init__(self, membrane_time_constant=0.005, threshold=0.5, reset_potential=0.0):
        """
        A simple integrate-and-fire model that converts the ribbon synapse release rate
        into a spike train.

        Args:
            membrane_time_constant (float): Integration time constant (s).
            threshold (float): Voltage threshold for spike generation.
            reset_potential (float): Membrane potential to reset to after a spike.

        Research Basis:
            - Basic integrate-and-fire models are widely used in auditory neuroscience to model
              temporal encoding (see, e.g., Joris et al., 2004).
        """
        self.tau = membrane_time_constant
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.V = 0.0
        self.spike_train = []

    def update(self, release_rate, dt):
        """
        Update the membrane potential based on the release rate and generate spikes if the threshold is exceeded.

        Args:
            release_rate (float): Input from the ribbon synapse (vesicles/s, scaled).
            dt (float): Time step (s).

        Returns:
            int: 1 if a spike is generated in this time step, else 0.
        """
        # Simple Euler integration.
        dV = (-self.V + release_rate) * (dt / self.tau)
        self.V += dV
        if self.V >= self.threshold:
            spike = 1
            self.V = self.reset_potential  # reset after spiking
        else:
            spike = 0
        self.spike_train.append(spike)
        return spike


class EfferentModulator:
    def __init__(self, baseline_gain=2.0, modulation_strength=0.5, min_gain=1.0):
        """
        Efferent Modulator for the cochlea, simulating the medial olivocochlear (MOC) efferent system.

        This class modulates the amplification factor of outer hair cells (OHCs) based on efferent activity.
        Higher efferent input reduces the OHC gain, thereby protecting the cochlea and contributing to dynamic range control.

        Args:
            baseline_gain (float): The default OHC amplification factor without efferent modulation.
            modulation_strength (float): The maximum reduction in gain due to efferent input.
            min_gain (float): The minimum allowable gain to ensure some amplification remains.

        Research Basis:
            - Guinan, J. J. Jr. (2006): Reviews the role of the medial olivocochlear system in modulating OHC function.
            - Liberman, M. C. (1991): Describes efferent influences on cochlear gain and auditory nerve responses.
        """
        self.baseline_gain = baseline_gain
        self.modulation_strength = modulation_strength
        self.min_gain = min_gain
        self.current_gain = baseline_gain

    def modulate(self, efferent_signal):
        """
        Modulate the OHC amplification factor based on a normalized efferent input signal.

        Args:
            efferent_signal (float): A normalized signal (0 to 1) representing efferent activity.
                                      A value of 1 corresponds to maximal efferent activation.

        Returns:
            float: The updated OHC gain after efferent modulation.
        """
        # Calculate the new gain by reducing the baseline gain proportionally to the efferent signal.
        self.current_gain = self.baseline_gain - self.modulation_strength * efferent_signal
        # Ensure the gain does not drop below the minimum gain.
        self.current_gain = max(self.current_gain, self.min_gain)
        return self.current_gain
