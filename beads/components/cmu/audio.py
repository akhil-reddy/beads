import sounddevice as sd
import numpy as np

from beads.core.cmu.sequencing.receive.audio import OuterEar

FS = 44100
ear_model = OuterEar(fs=FS, realtime=True)


def get_stimulus_continuous(indata, frames, time, status):
    if status:
        print(status)

    raw_audio = indata[:, 0] + 1j * indata[:, 1]

    processed_pressure = ear_model.function(raw_audio, input_is_pa=False, ref_db=94.0)

    peak_pa = np.max(np.abs(processed_pressure))
    print(f"Peak Pressure: {peak_pa:.2f} Pa", end="\r")

    outdata = processed_pressure.real + 1j * processed_pressure.imag


if __name__ == "__main__":
    try:
        with sd.Stream(samplerate=FS, channels=2, callback=get_stimulus_continuous):
            print("Listening...Press Ctrl+C to stop")
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("Stopped listening.")
