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
