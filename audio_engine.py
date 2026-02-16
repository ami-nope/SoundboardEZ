from __future__ import annotations

import sounddevice as sd
import numpy as np

from soundboard import Soundboard


class AudioEngine:
    def __init__(
        self,
        samplerate: int = 48000,
        blocksize: int = 512,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
        input_channels: int = 1,
        output_channels: int = 1,
        sounds_dir: str = "sounds",
    ) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.input_device = input_device
        self.output_device = output_device
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.soundboard = Soundboard(sounds_dir=sounds_dir, samplerate=samplerate)
        self.running = False

    def _audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            print(status)

        # Route mic to virtual cable first.
        outdata.fill(0.0)
        in_ch = indata.shape[1]
        out_ch = outdata.shape[1]
        ch = min(in_ch, out_ch)
        outdata[:, :ch] = indata[:, :ch]

        # Mix soundboard clips on top in-place.
        self.soundboard.mix_into(outdata)

        np.clip(outdata, -1.0, 1.0, out=outdata)

    def _resolve_output_device(self) -> str | int | None:
        if self.output_device is not None:
            return self.output_device

        # Typical VB-Cable playback endpoint name.
        for device in sd.query_devices():
            if device["max_output_channels"] > 0 and "CABLE Input" in device["name"]:
                return device["name"]
        return None

    def _resolve_input_device(self) -> str | int | None:
        if self.input_device is not None:
            return self.input_device
        return None

    def setup_soundboard(self, auto_hotkeys: bool = True) -> dict[str, str]:
        self.soundboard.load_directory()
        if auto_hotkeys:
            return self.soundboard.bind_hotkeys_auto()
        return {}

    def start(self) -> None:
        self.running = True

        input_device = self._resolve_input_device()
        output_device = self._resolve_output_device()

        if output_device is None:
            raise RuntimeError(
                "Could not find VB-Cable output device. Ensure 'CABLE Input' is installed and enabled."
            )

        with sd.Stream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype="float32",
            channels=(self.input_channels, self.output_channels),
            device=(input_device, output_device),
            callback=self._audio_callback,
        ):
            print("Audio engine running. Press Ctrl+C to stop.")
            while self.running:
                sd.sleep(1000)

    def stop(self) -> None:
        self.running = False


if __name__ == "__main__":
    print("This module is not the UI entrypoint.")
    print("Run: .\\venv\\Scripts\\python main.py")
