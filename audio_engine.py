from __future__ import annotations

import threading
import time
from typing import Callable

import sounddevice as sd
import numpy as np

from noise_suppression import MicNoiseSuppressor
from soundboard import Soundboard


class AudioEngine:
    def __init__(
        self,
        samplerate: int = 48000,
        blocksize: int = 480,
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
        self._stream_lock = threading.Lock()
        self._noise_lock = threading.Lock()
        self._mic_noise_suppression_enabled = False
        self._mic_input_gain = 1.0
        self._mic_noise_suppressor = MicNoiseSuppressor(sample_rate=self.samplerate)
        self._last_stream_status_log = 0.0
        self._last_mix_error_log = 0.0

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
        value = max(0.0, min(2.0, float(gain)))
        with self._noise_lock:
            self._mic_input_gain = value
        return value

    def mic_input_gain(self) -> float:
        with self._noise_lock:
            return float(self._mic_input_gain)

    def set_soundboard_enabled(self, enabled: bool) -> bool:
        return self.soundboard.set_enabled(bool(enabled))

    def is_soundboard_enabled(self) -> bool:
        return self.soundboard.is_enabled()

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
        # DirectSound is the most reliable backend for this app's threaded
        # stream lifecycle on Windows; WASAPI can fail in worker threads.
        if "directsound" in host:
            return 0
        if "wasapi" in host:
            return 1
        if "mme" in host:
            return 2
        if "wdm-ks" in host:
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

        # Prefer a single backend to avoid duplicate endpoints shown by
        # multiple host APIs.
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

    @staticmethod
    def _is_feedback_prone_input_name(name: str) -> bool:
        lowered = str(name).lower()
        markers = (
            "cable output",
            "vb-cable",
            "vb-audio",
            "virtual cable",
            "virtual audio cable",
            "voicemeeter",
            "loopback",
            "stereo mix",
            "what u hear",
            "wave out mix",
            "speaker wave",
            "monitor of",
            "blackhole",
            "soundflower",
            "input (vb-audio point)",
            "output (vb-audio point)",
        )
        if any(marker in lowered for marker in markers):
            return True
        if "input (" in lowered and "speaker" in lowered:
            return True
        return False

    @staticmethod
    def _is_mic_like_input_name(name: str) -> bool:
        lowered = str(name).lower()
        markers = (
            "microphone",
            "mic",
            "headset",
            "line in",
            "line-in",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _device_max_channels(device_ref: str | int | None, kind: str) -> int:
        key = "max_input_channels" if kind == "input" else "max_output_channels"
        try:
            dev = sd.query_devices(device_ref)
            return max(0, int(dev.get(key, 0)))
        except Exception:
            return 0

    @staticmethod
    def _device_hostapi_idx(device_ref: str | int | None) -> int | None:
        if device_ref is None:
            return None
        try:
            dev = sd.query_devices(device_ref)
            host = dev.get("hostapi")
            if host is None:
                return None
            return int(host)
        except Exception:
            return None

    @staticmethod
    def _hosts_compatible(input_ref: str | int | None, output_ref: str | int | None) -> bool:
        in_host = AudioEngine._device_hostapi_idx(input_ref)
        out_host = AudioEngine._device_hostapi_idx(output_ref)
        if in_host is None or out_host is None:
            return True
        return in_host == out_host

    def _find_input_device_for_host(self, hostapi_idx: int | None, allow_feedback_inputs: bool = False) -> int | None:
        default_in = self._default_device_index("input")
        candidates: list[tuple[int, int, int, int]] = []
        for idx, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            name = str(dev.get("name", "")).strip()
            if not name:
                continue
            if "microsoft sound mapper" in name.lower():
                continue
            if not allow_feedback_inputs and self._is_feedback_prone_input_name(name):
                continue
            dev_host = int(dev.get("hostapi", -1))
            if hostapi_idx is not None and dev_host != hostapi_idx:
                continue
            default_penalty = 0 if default_in is not None and idx == default_in else 1
            mic_penalty = 0 if self._is_mic_like_input_name(name) else 1
            host_name = self._hostapi_name(dev_host)
            host_priority = self._hostapi_priority(host_name)
            candidates.append((default_penalty, mic_penalty, host_priority, idx))

        if not candidates:
            if not allow_feedback_inputs:
                return self._find_input_device_for_host(hostapi_idx, allow_feedback_inputs=True)
            return None
        candidates.sort()
        return candidates[0][3]

    def _stream_channel_attempts(
        self,
        input_device: str | int | None,
        output_device: str | int | None,
    ) -> list[tuple[int, int]]:
        max_in = self._device_max_channels(input_device, "input")
        max_out = self._device_max_channels(output_device, "output")
        if max_in <= 0 or max_out <= 0:
            return []

        requested_in = max(1, int(self.input_channels))
        requested_out = max(1, int(self.output_channels))
        attempts: list[tuple[int, int]] = []

        def add(in_ch: int, out_ch: int) -> None:
            if in_ch <= 0 or out_ch <= 0:
                return
            if in_ch > max_in or out_ch > max_out:
                return
            candidate = (int(in_ch), int(out_ch))
            if candidate not in attempts:
                attempts.append(candidate)

        base_in = min(requested_in, max_in)
        base_out = min(requested_out, max_out)
        add(base_in, base_out)
        add(1, 1)
        if max_out >= 2:
            add(base_in, 2)
            add(1, 2)
        if max_in >= 2:
            add(2, base_out)
            add(2, 1)
        if max_in >= 2 and max_out >= 2:
            add(2, 2)
        return attempts

    def _stream_blocksize_attempts(self) -> list[int]:
        attempts: list[int] = []
        requested = int(self.blocksize)
        if requested > 0:
            attempts.append(requested)
        if 480 not in attempts:
            attempts.append(480)
        if 0 not in attempts:
            attempts.append(0)
        return attempts

    def _audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            now = time.monotonic()
            if now - self._last_stream_status_log >= 1.5:
                self._last_stream_status_log = now
                print(f"Audio stream status: {status}")

        # Route mic to virtual cable first (optionally denoised).
        outdata.fill(0.0)
        in_ch = indata.shape[1]
        out_ch = outdata.shape[1]
        ch = min(in_ch, out_ch)
        mic_source = indata[:, :ch]
        mic_frame = mic_source

        with self._noise_lock:
            ns_enabled = bool(self._mic_noise_suppression_enabled)
            mic_gain = float(self._mic_input_gain)

        if ns_enabled and self.is_noise_suppression_available():
            try:
                dry_mic_frame = np.ascontiguousarray(mic_source, dtype=np.float32)
                denoised = self._mic_noise_suppressor.process_mic_frame(dry_mic_frame)
                mic_frame = self._stabilize_noise_suppression(dry_mic_frame, denoised)
            except Exception as exc:
                print(f"RNNoise process error: {exc}")
                with self._noise_lock:
                    self._mic_noise_suppression_enabled = False
                mic_frame = mic_source

        if mic_gain != 1.0:
            mic_frame = mic_frame * mic_gain

        if in_ch == out_ch:
            outdata[:, :out_ch] = mic_frame[:, :out_ch]
        elif in_ch == 1 and out_ch > 1:
            # Broadcast mono mic to all output channels to avoid one-sided output.
            outdata[:, :out_ch] = mic_frame[:, :1]
        elif in_ch > 1 and out_ch == 1:
            outdata[:, 0] = mic_frame.mean(axis=1)
        else:
            outdata[:, :ch] = mic_frame[:, :ch]

        # Mix soundboard clips on top in-place (bypasses mic denoiser).
        try:
            self.soundboard.mix_into(outdata)
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_mix_error_log >= 1.5:
                self._last_mix_error_log = now
                print(f"Soundboard mix error: {exc}")

        np.clip(outdata, -1.0, 1.0, out=outdata)

    @staticmethod
    def _stabilize_noise_suppression(dry_frame: np.ndarray, denoised_frame: np.ndarray) -> np.ndarray:
        dry = np.ascontiguousarray(dry_frame, dtype=np.float32)
        wet = np.ascontiguousarray(denoised_frame, dtype=np.float32)
        if dry.shape != wet.shape:
            return wet

        dry_power = float(np.mean(dry * dry))
        wet_power = float(np.mean(wet * wet))

        if dry_power > 1e-7:
            min_wet_power = dry_power * 0.2
            if wet_power <= 1e-12:
                wet = dry
            elif wet_power < min_wet_power:
                boost = (min_wet_power / wet_power) ** 0.5
                wet = wet * np.float32(min(boost, 4.0))

        return np.ascontiguousarray(np.clip(wet, -1.0, 1.0), dtype=np.float32)

    def _resolve_output_device(self, preferred_hostapi: int | None = None) -> str | int | None:
        if self.output_device is not None:
            return self.output_device

        # Typical VB-Cable playback endpoint name.
        candidates: list[tuple[int, int, int]] = []
        for idx, device in enumerate(sd.query_devices()):
            if int(device.get("max_output_channels", 0)) <= 0:
                continue
            name = str(device.get("name", ""))
            if "cable input" not in name.lower():
                continue
            hostapi_idx = int(device.get("hostapi", -1))
            host_name = self._hostapi_name(hostapi_idx)
            host_priority = self._hostapi_priority(host_name)
            host_penalty = 0 if preferred_hostapi is not None and hostapi_idx == preferred_hostapi else 1
            candidates.append((host_penalty, host_priority, idx))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][2]

    def _resolve_input_device(self, preferred_hostapi: int | None = None) -> str | int | None:
        if self.input_device is not None:
            return self.input_device
        if preferred_hostapi is not None:
            by_host = self._find_input_device_for_host(preferred_hostapi)
            if by_host is not None:
                return by_host
        fallback = self._find_input_device_for_host(None)
        if fallback is not None:
            return fallback
        return self._default_device_index("input")

    def get_route_summary(self) -> tuple[str, str]:
        preferred_output_host = self._device_hostapi_idx(self.input_device)
        output_ref = self._resolve_output_device(preferred_hostapi=preferred_output_host)
        preferred_input_host = self._device_hostapi_idx(output_ref)
        input_ref = self._resolve_input_device(preferred_hostapi=preferred_input_host)
        input_name = self._device_name(input_ref, "input")
        output_name = self._device_name(output_ref, "output")
        return input_name, output_name

    def setup_soundboard(self, auto_hotkeys: bool = True) -> dict[str, str]:
        self.soundboard.load_directory()
        if auto_hotkeys:
            return self.soundboard.bind_hotkeys_auto()
        return {}

    def start(self, on_started: Callable[[], None] | None = None) -> None:
        with self._stream_lock:
            if self.running:
                if on_started is not None:
                    try:
                        on_started()
                    except Exception:
                        pass
                return
            self.running = True
            self._last_stream_status_log = 0.0
            self._last_mix_error_log = 0.0

        preferred_output_host = self._device_hostapi_idx(self.input_device)
        output_device = self._resolve_output_device(preferred_hostapi=preferred_output_host)

        if output_device is None:
            raise RuntimeError(
                "Could not find VB-Cable output device. Ensure 'CABLE Input' is installed and enabled."
            )

        output_host = self._device_hostapi_idx(output_device)
        input_device = self._resolve_input_device(preferred_hostapi=output_host)

        if not self._hosts_compatible(input_device, output_device):
            in_name = self._device_name(input_device, "input")
            out_name = self._device_name(output_device, "output")
            raise RuntimeError(
                "Mic and mix output are on different host backends. "
                f"Mic='{in_name}' Output='{out_name}'. Select matching backend devices."
            )

        channel_attempts = self._stream_channel_attempts(input_device, output_device)
        if not channel_attempts:
            in_name = self._device_name(input_device, "input")
            out_name = self._device_name(output_device, "output")
            raise RuntimeError(
                "Selected audio devices do not expose required channels. "
                f"Mic='{in_name}' Output='{out_name}'."
            )
        blocksize_attempts = self._stream_blocksize_attempts()
        open_errors: list[str] = []

        try:
            for in_channels, out_channels in channel_attempts:
                for blocksize in blocksize_attempts:
                    try:
                        with sd.Stream(
                            samplerate=self.samplerate,
                            blocksize=blocksize,
                            dtype="float32",
                            channels=(in_channels, out_channels),
                            device=(input_device, output_device),
                            callback=self._audio_callback,
                        ):
                            if on_started is not None:
                                try:
                                    on_started()
                                except Exception:
                                    pass
                            while True:
                                with self._stream_lock:
                                    if not self.running:
                                        break
                                sd.sleep(120)
                        return
                    except Exception as exc:
                        open_errors.append(
                            f"in={in_channels}, out={out_channels}, blocksize={blocksize}: {exc}"
                        )
            if open_errors:
                raise RuntimeError(f"Error opening Stream: {open_errors[0]}")
            raise RuntimeError("Error opening Stream: no valid stream configuration.")
        finally:
            with self._stream_lock:
                self.running = False

    def stop(self) -> None:
        with self._stream_lock:
            self.running = False

    def shutdown(self) -> None:
        self.stop()
        with self._noise_lock:
            self._mic_noise_suppression_enabled = False
        try:
            self.soundboard.set_enabled(False)
        except Exception:
            pass
        try:
            self._mic_noise_suppressor.close()
        except Exception:
            pass
        try:
            self.soundboard.stop_all()
        except Exception:
            pass
        try:
            self.soundboard.clear_hotkeys()
        except Exception:
            pass


if __name__ == "__main__":
    print("This module is not the UI entrypoint.")
    print("Run: .\\venv\\Scripts\\python main.py")
