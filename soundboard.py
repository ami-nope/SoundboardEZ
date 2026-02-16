from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import threading
import wave

import keyboard
import numpy as np


@dataclass(frozen=True)
class Sound:
    name: str
    audio: np.ndarray  # shape: (frames, channels), dtype: float32, range: [-1, 1]


class Soundboard:
    def __init__(self, sounds_dir: str = "sounds", samplerate: int = 48000) -> None:
        self.sounds_dir = Path(sounds_dir)
        self.samplerate = samplerate
        self.sounds: dict[str, Sound] = {}
        self.hotkeys: dict[str, str] = {}  # hotkey -> sound name
        self._active: list[dict[str, object]] = []
        self._lock = threading.Lock()
        self._registered_hotkeys: list[str] = []
        self._volume = 1.0  # soundboard gain (0.0 = mute, 1.0 = unity)

    def set_volume(self, volume: float) -> None:
        clamped = max(0.0, min(2.0, float(volume)))
        with self._lock:
            self._volume = clamped

    def get_volume(self) -> float:
        with self._lock:
            return self._volume

    def load_wav(self, name: str, path: str | Path) -> None:
        audio, file_rate = _read_wav_float32(path)
        if file_rate != self.samplerate:
            audio = _resample_audio(audio, file_rate, self.samplerate)
        self.sounds[name] = Sound(name=name, audio=audio)

    def load_audio_file(self, name: str, path: str | Path) -> None:
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()
        if suffix == ".wav":
            self.load_wav(name, path_obj)
            return
        if suffix in {".mp3", ".ogg", ".m4a", ".aac", ".flac"}:
            audio = _decode_compressed_audio_to_float32(path_obj, self.samplerate, channels=2)
            self.sounds[name] = Sound(name=name, audio=audio)
            return
        raise ValueError(f"Unsupported audio format: {path_obj.suffix}")

    def load_directory(self) -> None:
        supported = {".wav", ".mp3", ".ogg", ".m4a", ".aac", ".flac"}
        for path in sorted(self.sounds_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in supported:
                self.load_audio_file(path.stem, path)

    def bind_hotkey(self, hotkey: str, sound_name: str) -> None:
        if sound_name not in self.sounds:
            raise KeyError(f"Unknown sound '{sound_name}'")
        self.hotkeys[hotkey] = sound_name

        handle = keyboard.add_hotkey(hotkey, self.trigger, args=(sound_name,))
        self._registered_hotkeys.append(handle)

    def bind_hotkeys_auto(self) -> dict[str, str]:
        default_keys = list("1234567890qwertyuiopasdfghjklzxcvbnm")
        mapping: dict[str, str] = {}
        for key, sound_name in zip(default_keys, sorted(self.sounds.keys())):
            self.bind_hotkey(key, sound_name)
            mapping[key] = sound_name
        return mapping

    def clear_hotkeys(self) -> None:
        for handle in self._registered_hotkeys:
            keyboard.remove_hotkey(handle)
        self._registered_hotkeys.clear()
        self.hotkeys.clear()

    def trigger(self, sound_name: str) -> None:
        sound = self.sounds.get(sound_name)
        if sound is None:
            return
        with self._lock:
            self._active.append({"audio": sound.audio, "pos": 0})

    def mix_into(self, outdata: np.ndarray) -> None:
        with self._lock:
            if not self._active:
                return

            frames, out_channels = outdata.shape
            still_active: list[dict[str, object]] = []
            volume = self._volume

            for playback in self._active:
                audio = playback["audio"]  # type: ignore[assignment]
                pos = playback["pos"]  # type: ignore[assignment]
                audio = audio  # type: ignore[no-redef]
                pos = int(pos)

                remaining = audio.shape[0] - pos
                if remaining <= 0:
                    continue

                n = min(frames, remaining)
                chunk = audio[pos : pos + n]
                _mix_chunk_to_channels(outdata[:n], chunk, out_channels, volume)

                pos += n
                if pos < audio.shape[0]:
                    playback["pos"] = pos
                    still_active.append(playback)

            self._active = still_active


def _mix_chunk_to_channels(out_chunk: np.ndarray, src_chunk: np.ndarray, out_channels: int, volume: float = 1.0) -> None:
    src_channels = src_chunk.shape[1]
    if src_channels == out_channels:
        out_chunk += src_chunk * volume
        return

    if src_channels == 1 and out_channels > 1:
        out_chunk += np.repeat(src_chunk, out_channels, axis=1) * volume
        return

    if src_channels > 1 and out_channels == 1:
        out_chunk[:, 0] += src_chunk.mean(axis=1) * volume
        return

    ch = min(src_channels, out_channels)
    out_chunk[:, :ch] += src_chunk[:, :ch] * volume


def _read_wav_float32(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)

    audio = _pcm_bytes_to_float32(raw, sample_width, channels)
    return audio, sample_rate


def _pcm_bytes_to_float32(raw: bytes, sample_width: int, channels: int) -> np.ndarray:
    if sample_width == 1:
        pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        pcm = (pcm - 128.0) / 128.0
    elif sample_width == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (
            data[:, 0].astype(np.int32)
            | (data[:, 1].astype(np.int32) << 8)
            | (data[:, 2].astype(np.int32) << 16)
        )
        sign = 1 << 23
        ints = (ints ^ sign) - sign
        pcm = ints.astype(np.float32) / float(1 << 23)
    elif sample_width == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    return pcm.reshape(-1, channels).astype(np.float32, copy=False)


def _resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio

    src_len = audio.shape[0]
    if src_len == 0:
        return audio

    dst_len = int(round(src_len * (dst_rate / src_rate)))
    if dst_len <= 1:
        return audio[:1]

    x_src = np.linspace(0.0, 1.0, src_len, endpoint=False)
    x_dst = np.linspace(0.0, 1.0, dst_len, endpoint=False)

    out = np.empty((dst_len, audio.shape[1]), dtype=np.float32)
    for ch in range(audio.shape[1]):
        out[:, ch] = np.interp(x_dst, x_src, audio[:, ch]).astype(np.float32)
    return out


def _decode_compressed_audio_to_float32(path: Path, samplerate: int, channels: int = 2) -> np.ndarray:
    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg not found; cannot decode compressed audio.")

    cmd = [
        ffmpeg_exe,
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(samplerate),
        "-ac",
        str(channels),
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for '{path.name}'.")

    pcm = np.frombuffer(result.stdout, dtype=np.float32)
    if pcm.size == 0:
        raise RuntimeError(f"Decoded audio is empty: '{path.name}'.")
    frames = pcm.size // channels
    return pcm[: frames * channels].reshape(frames, channels)


def _get_ffmpeg_exe() -> str | None:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore

        return get_ffmpeg_exe()
    except Exception:
        return None
