import numpy as np
from scipy.signal import iirnotch, butter, sosfilt, sosfiltfilt


class Pinna:
    """
    Pinna spectral shaping: apply multiple configurable notches and peaks.
    Notches use `iirnotch` (good numeric behavior). Peaks use 2nd-order bandpass SOS scaled by gain.
    realtime=True -> use causal filtering (sosfilt). realtime=False -> zero-phase (sosfiltfilt).
    """

    def __init__(self, fs, notch_freqs=None, notch_depths=None,
                 peak_freqs=None, peak_gains=None, Q=10, realtime=True):
        self.fs = float(fs)
        notch_freqs = notch_freqs or []
        notch_depths = notch_depths or [10.0] * len(notch_freqs)
        peak_freqs = peak_freqs or []
        peak_gains = peak_gains or [6.0] * len(peak_freqs)
        self.Q = Q
        self.realtime = bool(realtime)

        # Precompute SOS for peaks and (b,a) for notches
        self.notches = []
        for f0, depth_db in zip(notch_freqs, notch_depths):
            # design notch: returns (b,a)
            w0 = float(f0)
            b, a = iirnotch(w0 / (self.fs / 2.0), Q=self.Q)
            # we'll apply depth by mixing the filtered output with identity:
            # y_out = gain * x + (1-gain) * y_notched where gain = 10^(-depth/20)
            gain = 10 ** (-depth_db / 20.0)
            self.notches.append((b, a, gain))

        self.peaks = []
        for f0, gain_db in zip(peak_freqs, peak_gains):
            # design 2nd-order bandpass sos around f0 with bandwidth from Q
            bw = f0 / self.Q
            low = max(1.0, f0 - bw / 2.0)
            high = min(self.fs / 2.0 - 1.0, f0 + bw / 2.0)
            if low <= 0:
                low = f0 * 0.9
            # 2nd-order butterband in SOS form
            sos = butter(N=2, Wn=[low, high], btype='bandpass', fs=self.fs, output='sos')
            gain_lin = 10 ** (gain_db / 20.0)
            self.peaks.append((sos, gain_lin))

    def function(self, x):
        y = np.asarray(x, dtype=float)
        # apply notches: y = gain * x + (1 - gain) * notch(y)
        for b, a, gain in self.notches:
            # causal vs zero-phase: use lfilter via sos (convert b,a -> sos is fine but simpler to use sosfilt)
            # convert b,a -> sos (2nd-order sections) for stable filtering if needed
            # we'll use lfilter style via sosfilt by converting b,a to sos numerically:
            # (for simplicity, use np.copy filtering by applying lfilter with (b,a) implemented by sosfilt)
            # but scipy doesn't provide direct b,a -> sos conversion easily; here we call lfilter via b,a
            # Use scipy.signal.sosfiltfilt for bandpass; for notch it's fine to do lfilter with b,a
            from scipy.signal import lfilter
            y_notch = lfilter(b, a, y)
            y = gain * x + (1.0 - gain) * y_notch  # mix original & notched

        # apply peaks: cascade SOS bandpass scaled by gain
        for sos, gain_lin in self.peaks:
            if self.realtime:
                y = sosfilt(sos, y)
            else:
                # zero-phase for offline analysis
                y = sosfiltfilt(sos, y)
            # apply gain (multiply output, preserving DC scaling)
            y = y * gain_lin
        return y


class EarCanal:
    """
    Ear canal resonance: approximate quarter-wave resonance around f0 = c / (4 L).
    Implemented as bandpass (SOS), with optional realtime / zero-phase option.
    """

    def __init__(self, fs, length_m=0.025, gain_db=15.0, Q=4.0, realtime=True):
        self.fs = float(fs)
        c = 343.0
        f0 = c / (4.0 * float(length_m))
        self.f0 = float(f0)
        self.gain_db = float(gain_db)
        self.realtime = bool(realtime)
        # bandwidth set via Q (f0/Q)
        bw = self.f0 / Q
        low = max(1.0, self.f0 - bw / 2.0)
        high = min(self.fs / 2.0 - 1.0, self.f0 + bw / 2.0)
        self.sos = butter(2, [low, high], btype='bandpass', fs=self.fs, output='sos')
        self.gain_lin = 10 ** (gain_db / 20.0)

    def function(self, x):
        if self.realtime:
            y = sosfilt(self.sos, x)
        else:
            y = sosfiltfilt(self.sos, x)
        return y * self.gain_lin


class OuterEar:
    def __init__(self, fs, realtime=True):
        notch_freqs = [6000.0, 8000.0]
        notch_depths = [10.0, 8.0]
        peak_freqs = [4000.0, 12000.0]
        peak_gains = [6.0, 4.0]
        self.pinna = Pinna(fs, notch_freqs, notch_depths, peak_freqs, peak_gains, Q=8, realtime=realtime)
        self.canal = EarCanal(fs, length_m=0.025, gain_db=12.0, Q=4.0, realtime=realtime)

    def function(self, x, input_is_pa=False, ref_db=94.0, dtype=np.float32):
        """
        Process waveform through outer-ear chain and return pressure (Pa).

        Args:
            x: 1D numpy array, digital waveform. If input_is_pa==True then `x` is already in Pascals.
            input_is_pa: bool, whether `x` is already in Pa.
            ref_db: float, dB SPL value that maps digital amplitude 1.0 -> ref_db SPL. (default 94 dB -> 1 Pa)
            dtype: output dtype for the pressure array (float32 recommended).

        Returns:
            pressure: numpy array (same length as x) in Pascals (dtype dtype).
        """
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("expected 1D waveform (mono).")

        # Filter as double precision for numeric stability
        x64 = x.astype(np.float64)

        # apply pinna + canal (these functions should also be stable in float64)
        y64 = self.canal.function(self.pinna.function(x64))

        # if caller already gave us Pa, return directly (no scaling)
        if input_is_pa:
            return y64.astype(dtype)

        # map digital full-scale 1.0 -> ref_db SPL -> ref_pa (Pa)
        # ref_pa = 20 µPa * 10^(ref_db/20)
        ref_pa = 20e-6 * (10.0 ** (ref_db / 20.0))

        # scale waveform to Pa. This assumes your digital amplitude 1.0 corresponds to ref_db SPL
        pressure = (y64 * ref_pa).astype(dtype)

        return pressure

# TODO: Temporary code block to test these cells. Input and output should be through files (which can be used for the demo)

