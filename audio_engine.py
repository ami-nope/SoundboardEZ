from __future__ import annotations

import threading

import sounddevice as sd
import numpy as np

from noise_suppression import MicNoiseSuppressor
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
        self._noise_lock = threading.Lock()
        self._mic_noise_suppression_enabled = False
        self._mic_input_gain = 0.5
        self._mic_noise_suppressor = MicNoiseSuppressor(sample_rate=self.samplerate)

    def is_noise_suppression_available(self) -> bool:
        return bool(self._mic_noise_suppressor.available)

    def is_noise_suppression_enabled(self) -> bool:
        with self._noise_lock:
            return bool(self._mic_noise_suppression_enabled)

    def noise_suppression_error(self) -> str | None:
        return self._mic_noise_suppressor.error_message

    def noise_suppression_backend(self) -> str:
        if self._mic_noise_suppressor.backend_name:
            return self._mic_noise_suppressor.backend_name
        return "unavailable"

    def set_noise_suppression_enabled(self, enabled: bool) -> bool:
        target = bool(enabled)
        with self._noise_lock:
            if target and not self.is_noise_suppression_available():
                self._mic_noise_suppression_enabled = False
                return False
            self._mic_noise_suppression_enabled = target
            if target:
                self._mic_noise_suppressor.reset()
            return self._mic_noise_suppression_enabled

    def set_mic_input_gain(self, gain: float) -> float:
        value = max(0.0, min(1.0, float(gain)))
        with self._noise_lock:
            self._mic_input_gain = value
        return value

    def mic_input_gain(self) -> float:
        with self._noise_lock:
            return float(self._mic_input_gain)

    @staticmethod
    def _hostapi_name(hostapi_idx: int) -> str:
        try:
            hostapi = sd.query_hostapis(hostapi_idx)
            return str(hostapi.get("name", ""))
        except Exception:
            return ""

    @staticmethod
    def _device_name_key(name: str) -> str:
        return " ".join(str(name).split()).strip().lower()

    @staticmethod
    def _hostapi_priority(host_name: str) -> int:
        host = host_name.lower()
        if "wasapi" in host:
            return 0
        if "wdm-ks" in host:
            return 1
        if "directsound" in host:
            return 2
        if "mme" in host:
            return 3
        return 4

    def _list_devices(self, kind: str) -> list[tuple[int, str]]:
        if kind not in {"input", "output"}:
            return []

        channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
        candidates: list[tuple[int, str, int]] = []

        for idx, dev in enumerate(sd.query_devices()):
            max_channels = int(dev.get(channel_key, 0))
            if max_channels <= 0:
                continue

            name = str(dev.get("name", "")).strip()
            if not name:
                continue
            if "microsoft sound mapper" in name.lower():
                continue

            host = self._hostapi_name(int(dev.get("hostapi", 0)))
            priority = self._hostapi_priority(host)
            candidates.append((idx, name, priority))

        if not candidates:
            return []

        # Prefer a single backend (WASAPI first) to avoid duplicate endpoints
        # shown by multiple host APIs.
        best_priority = min(priority for _, _, priority in candidates)
        candidates = [row for row in candidates if row[2] == best_priority]

        selected: dict[str, tuple[int, str]] = {}
        for idx, name, _ in candidates:
            key = self._device_name_key(name)
            current = selected.get(key)
            if current is None:
                selected[key] = (idx, name)
                continue
            current_idx, current_name = current
            if len(name) > len(current_name):
                selected[key] = (idx, name)
                continue
            if len(name) == len(current_name) and idx < current_idx:
                selected[key] = (idx, name)

        devices = list(selected.values())
        devices.sort(key=lambda row: row[1].lower())
        return devices

    def list_input_devices(self) -> list[tuple[int, str]]:
        return self._list_devices("input")

    def list_output_devices(self) -> list[tuple[int, str]]:
        return self._list_devices("output")

    @staticmethod
    def _default_device_index(kind: str) -> int | None:
        try:
            defaults = sd.default.device
            if defaults is None:
                return None
            if kind == "input":
                idx = defaults[0]
            else:
                idx = defaults[1]
            if idx is None:
                return None
            idx = int(idx)
            if idx < 0:
                return None
            return idx
        except Exception:
            return None

    @staticmethod
    def _device_name(device_ref: str | int | None, kind: str) -> str:
        if device_ref is None:
            idx = AudioEngine._default_device_index(kind)
            if idx is None:
                return "System Default"
            try:
                return str(sd.query_devices(idx).get("name", "System Default"))
            except Exception:
                return "System Default"
        try:
            return str(sd.query_devices(device_ref).get("name", str(device_ref)))
        except Exception:
            return str(device_ref)

    def _audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            print(status)

        # Route mic to virtual cable first (optionally denoised).
        outdata.fill(0.0)
        in_ch = indata.shape[1]
        out_ch = outdata.shape[1]
        ch = min(in_ch, out_ch)
        mic_frame = indata[:, :ch]

        with self._noise_lock:
            ns_enabled = bool(self._mic_noise_suppression_enabled)
            mic_gain = float(self._mic_input_gain)

        if ns_enabled and self.is_noise_suppression_available():
            try:
                mic_frame = self._mic_noise_suppressor.process_mic_frame(mic_frame)
            except Exception as exc:
                print(f"RNNoise process error: {exc}")
                with self._noise_lock:
                    self._mic_noise_suppression_enabled = False
                mic_frame = indata[:, :ch]

        if mic_gain != 1.0:
            mic_frame = mic_frame * mic_gain

        outdata[:, :ch] = mic_frame

        # Mix soundboard clips on top in-place (bypasses mic denoiser).
        self.soundboard.mix_into(outdata)

        np.clip(outdata, -1.0, 1.0, out=outdata)

    def _resolve_output_device(self) -> str | int | None:
        if self.output_device is not None:
            return self.output_device

        # Typical VB-Cable playback endpoint name.
        for idx, device in enumerate(sd.query_devices()):
            if device["max_output_channels"] > 0 and "CABLE Input" in device["name"]:
                return idx
        return None

    def _resolve_input_device(self) -> str | int | None:
        if self.input_device is not None:
            return self.input_device
        return self._default_device_index("input")

    def get_route_summary(self) -> tuple[str, str]:
        input_ref = self._resolve_input_device()
        output_ref = self._resolve_output_device()
        input_name = self._device_name(input_ref, "input")
        output_name = self._device_name(output_ref, "output")
        return input_name, output_name

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
