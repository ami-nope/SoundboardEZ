from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import parse_qs, quote_plus, urlparse
import wave

import numpy as np
import requests
import sounddevice as sd
from PyQt6.QtCore import (
    QEvent,
    QEasingCurve,
    QLockFile,
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    QPropertyAnimation,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QAction, QColor, QBrush, QFont, QFontMetrics, QLinearGradient, QPainter, QPainterPath, QPen, QRegion
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMainWindow,
    QMessageBox,
    QMenu,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QInputDialog,
    QScrollArea,
    QSpacerItem,
    QSizePolicy,
    QSlider,
    QStyle,
    QStyledItemDelegate,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from audio_engine import AudioEngine
from app_state import AppState, load_app_state, save_app_state
from scraper import MYINSTANTS_INDEX_URL, download_via_dotnet, fetch_myinstants_sounds_page
from startup_manager import STARTUP_ARG, is_startup_enabled, set_startup_enabled
from update_checker import UpdateInfo, check_for_update
from updater import download_file, download_delta_files, launch_apply_and_exit
from version import APP_VERSION

PHI = 1.618
SKIP_UPDATE_ONCE_ARG = "--skip-update-once"
FEED_BUTTON_STYLE = """
    QPushButton {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 52);
        font-size: 13px;
        font-weight: 600;
        min-height: 0px;
        max-height: 32px;
        padding: 0px 10px;
        color: #eaf2fb;
        background: rgba(64, 82, 108, 220);
    }
    QPushButton:hover {
        background: rgba(81, 101, 130, 230);
        border-color: rgba(96, 214, 255, 188);
    }
    QPushButton:pressed {
        background: rgba(58, 77, 102, 235);
    }
    QPushButton[active="true"] {
        background: #0f6f98;
        border-color: #62d6ff;
        color: #f7fbff;
    }
    QPushButton:disabled {
        background: rgba(27, 39, 58, 220);
        border-color: rgba(71, 85, 105, 120);
        color: #8fa5c1;
    }
"""


@dataclass(frozen=True)
class RemoteSoundItem:
    name: str
    url: str


@dataclass
class RuntimeVolumeState:
    preview_gain: float = 0.08
    mic_gain: float = 0.6
    soundboard_gain: float = 0.15
    speaker_gain: float = 0.02


class ManagedOutputPlayer:
    """Plays audio to an output device using a callback-driven stream.

    All potentially slow PortAudio operations (open / start / stop / close)
    are performed on a private background thread so the caller (usually the
    Qt UI thread) never blocks.
    """

    def __init__(self, gain: float = 1.0) -> None:
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._active = False
        self._gain = max(0.0, min(2.0, float(gain)))

    def set_gain(self, gain: float) -> None:
        value = max(0.0, min(2.0, float(gain)))
        with self._lock:
            self._gain = value

    def is_active(self) -> bool:
        with self._lock:
            return bool(self._active and self._stream is not None)

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def close(self) -> None:
        self.stop()

    def _stop_locked(self) -> None:
        stream = self._stream
        self._stream = None
        self._active = False
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def play(
        self,
        audio: np.ndarray,
        samplerate: int,
        device: str | int | None = None,
    ) -> None:
        """Start playback.  The heavy PortAudio work runs off-thread."""
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
            raise ValueError("ManagedOutputPlayer requires non-empty 1D/2D audio array.")
        arr = np.ascontiguousarray(np.clip(arr, -1.0, 1.0), dtype=np.float32)
        channels = int(arr.shape[1])

        # Mark as active; actual stop + open happen on the daemon thread
        # so no PortAudio call ever blocks the caller.
        with self._lock:
            self._active = True

        def _open_and_start() -> None:
            # Stop the previous stream *on this background thread* so
            # Pa_StopStream / Pa_CloseStream never stall the UI.
            with self._lock:
                self._stop_locked()
                self._active = True  # Re-arm after stop clears the flag
            try:
                position = 0

                def callback(outdata, frames, _time, _status):
                    nonlocal position
                    end = min(position + frames, arr.shape[0])
                    n = end - position
                    outdata.fill(0.0)
                    if n > 0:
                        out_view = outdata[:n, :channels]
                        out_view[:] = arr[position:end]
                        with self._lock:
                            gain_now = float(self._gain)
                        if gain_now != 1.0:
                            out_view *= np.float32(gain_now)
                            np.clip(out_view, -1.0, 1.0, out=out_view)
                    position = end
                    if position >= arr.shape[0]:
                        raise sd.CallbackStop

                holder: dict[str, sd.OutputStream | None] = {"stream": None}

                def finished_callback() -> None:
                    stream_ref = holder["stream"]
                    if stream_ref is not None:
                        try:
                            stream_ref.close()
                        except Exception:
                            pass
                    with self._lock:
                        self._active = False
                        if self._stream is holder["stream"]:
                            self._stream = None

                stream = sd.OutputStream(
                    samplerate=int(samplerate),
                    channels=channels,
                    dtype="float32",
                    device=device,
                    callback=callback,
                    finished_callback=finished_callback,
                )
                holder["stream"] = stream
                with self._lock:
                    # If stop() was called while we were opening, bail out.
                    if not self._active:
                        try:
                            stream.close()
                        except Exception:
                            pass
                        return
                    self._stream = stream
                stream.start()
            except Exception as exc:
                with self._lock:
                    self._active = False
                print(f"ManagedOutputPlayer stream error: {exc}")

        threading.Thread(target=_open_and_start, daemon=True).start()


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in (" ", "-", "_")) else "_" for ch in name).strip()
    cleaned = "_".join(cleaned.split())
    return cleaned[:80] or "sound"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _is_wav_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".wav")


def _url_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix


def _download_to_path_streaming(
    url: str,
    dst_path: Path,
    timeout: float = 20.0,
    progress_cb=None,
    chunk_size: int = 256 * 1024,
) -> tuple[int, int | None]:
    downloaded = 0
    total_size: int | None = None
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Try Python requests first; if TLS fails, fall back to .NET on Windows.
    try:
        with requests.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            content_length = str(response.headers.get("content-length", "")).strip()
            if content_length.isdigit():
                total_size = int(content_length)
            if callable(progress_cb):
                progress_cb(downloaded, total_size)
            with dst_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=max(32 * 1024, int(chunk_size))):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if callable(progress_cb):
                        progress_cb(downloaded, total_size)
        return downloaded, total_size
    except (requests.ConnectionError, requests.exceptions.SSLError):
        # TLS handshake rejected (Cloudflare) — use .NET fallback on Windows.
        if sys.platform != "win32":
            raise

    # .NET fallback: download the entire file, then report final size.
    dst_path.unlink(missing_ok=True)
    if callable(progress_cb):
        progress_cb(0, None)
    download_via_dotnet(url, str(dst_path), timeout=timeout)
    if not dst_path.exists():
        raise RuntimeError("Download via .NET produced no file")
    file_size = dst_path.stat().st_size
    if callable(progress_cb):
        progress_cb(file_size, file_size)
    return file_size, file_size


def _ffmpeg_exists() -> bool:
    return _get_ffmpeg_exe() is not None


def _convert_to_wav_ffmpeg(src: Path, dst: Path) -> bool:
    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is None:
        return False
    cmd = [ffmpeg_exe, "-y", "-v", "error", "-i", str(src), str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and dst.exists()


def _decode_audio_for_preview(path: Path) -> tuple[np.ndarray, int]:
    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is not None:
        sr = 48000
        ch = 2
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
            str(sr),
            "-ac",
            str(ch),
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and result.stdout:
            pcm = np.array(np.frombuffer(result.stdout, dtype=np.float32), copy=True)
            frames = pcm.size // ch
            if frames > 0:
                return np.ascontiguousarray(pcm[: frames * ch].reshape(frames, ch), dtype=np.float32), sr

    if path.suffix.lower() == ".wav":
        return _read_wav_float32(path)

    raise RuntimeError("Preview decode failed. Install ffmpeg for non-WAV preview.")


def _read_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)

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
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    return pcm.reshape(-1, channels).astype(np.float32, copy=False), sample_rate


def _resolve_myinstants_feed_url(raw: str) -> tuple[str | None, str | None]:
    text = raw.strip()
    if not text:
        return MYINSTANTS_INDEX_URL, None

    if text.startswith(("http://", "https://")):
        parsed = urlparse(text)
        host = (parsed.netloc or "").lower()
        if "myinstants.com" in host:
            return text, None
        if "google." in host:
            q = parse_qs(parsed.query).get("q", [""])[0].strip()
            if not q:
                return None, "Google link has no query."
            if "myinstants.com" in q and q.startswith(("http://", "https://")):
                nested = urlparse(q)
                nested_host = (nested.netloc or "").lower()
                if "myinstants.com" in nested_host:
                    return q, None
            return f"https://www.myinstants.com/en/search/?name={quote_plus(q)}", None
        return None, "Only myinstants.com or Google links are allowed."

    # Plain tag/search text.
    return f"https://www.myinstants.com/en/search/?name={quote_plus(text)}", None


def _get_ffmpeg_exe() -> str | None:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore

        return get_ffmpeg_exe()
    except Exception:
        return None


def _runtime_install_dir() -> Path:
    if getattr(sys, "frozen", False):
        try:
            return Path(sys.executable).resolve().parent
        except Exception:
            pass
    return Path(__file__).resolve().parent


def _default_user_sounds_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "SoundboardEZ" / "sounds"
    return Path.home() / "AppData" / "Local" / "SoundboardEZ" / "sounds"


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    probe = path / f".sbz_write_probe_{os.getpid()}_{int(time.time() * 1000)}.tmp"
    try:
        with probe.open("wb") as fh:
            fh.write(b"ok")
        return True
    except Exception:
        return False
    finally:
        try:
            probe.unlink(missing_ok=True)
        except Exception:
            pass


def _seed_runtime_sounds_if_empty(target_dir: Path, source_dir: Path) -> None:
    try:
        if any(target_dir.iterdir()):
            return
    except Exception:
        return
    if not source_dir.is_dir():
        return
    supported = {".wav", ".mp3", ".ogg", ".m4a", ".aac", ".flac"}
    for src in sorted(source_dir.iterdir()):
        if not src.is_file() or src.suffix.lower() not in supported:
            continue
        dst = target_dir / src.name
        if dst.exists():
            continue
        try:
            shutil.copy2(src, dst)
        except Exception:
            continue


def _resolve_runtime_sounds_dir() -> Path:
    install_sounds = _runtime_install_dir() / "sounds"
    user_sounds = _default_user_sounds_dir()
    fallback = Path(tempfile.gettempdir()) / "SoundboardEZ" / "sounds"
    if getattr(sys, "frozen", False):
        candidates = [user_sounds, install_sounds, fallback]
    else:
        candidates = [Path("sounds"), user_sounds, fallback]

    for candidate in candidates:
        if not _is_writable_dir(candidate):
            continue
        if candidate == user_sounds:
            _seed_runtime_sounds_if_empty(candidate, install_sounds)
        return candidate
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


class FetchWorker(QObject):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, url: str, page: int) -> None:
        super().__init__()
        self.url = url
        self.page = page

    def run(self) -> None:
        try:
            rows = fetch_myinstants_sounds_page(self.url, page=self.page)
            self.finished.emit(rows)
        except Exception as exc:
            self.error.emit(str(exc))


class ImportWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(object)

    def __init__(self, selected: list[RemoteSoundItem], sounds_dir: Path) -> None:
        super().__init__()
        self.selected = selected
        self.sounds_dir = sounds_dir

    def run(self) -> None:
        try:
            self.sounds_dir.mkdir(parents=True, exist_ok=True)
            imported: list[str] = []
            skipped: list[str] = []
            ffmpeg_ready = _ffmpeg_exists()
            existing_basenames = {p.stem.lower() for p in self.sounds_dir.iterdir() if p.is_file()}
            batch_basenames: set[str] = set()
            total_items = max(1, len(self.selected))

            for idx, item in enumerate(self.selected, start=1):
                self.progress.emit(
                    {
                        "phase": "item-start",
                        "item_name": item.name,
                        "item_index": idx,
                        "total_items": total_items,
                    }
                )
                base = _safe_name(item.name)
                base_key = base.lower()
                if base_key in existing_basenames or base_key in batch_basenames:
                    skipped.append(f"{item.name} (duplicate name)")
                    continue
                suffix = _url_suffix(item.url)
                if suffix not in {".wav", ".mp3"}:
                    suffix = ".wav"
                out_path = self.sounds_dir / f"{base}{suffix}"

                last_emit = [0.0]

                def progress_cb(downloaded: int, total: int | None) -> None:
                    now = time.monotonic()
                    if total is not None and downloaded < total and now - last_emit[0] < 0.08:
                        return
                    last_emit[0] = now
                    self.progress.emit(
                        {
                            "phase": "download",
                            "item_name": item.name,
                            "item_index": idx,
                            "total_items": total_items,
                            "downloaded_bytes": int(downloaded),
                            "total_bytes": int(total) if total is not None else None,
                        }
                    )

                direct_suffix = _url_suffix(item.url)
                if direct_suffix in {".wav", ".mp3"}:
                    try:
                        _download_to_path_streaming(item.url, out_path, progress_cb=progress_cb)
                    except Exception as exc:
                        out_path.unlink(missing_ok=True)
                        skipped.append(f"{item.name} ({exc})")
                        continue
                    imported.append(str(out_path))
                    existing_basenames.add(base_key)
                    batch_basenames.add(base_key)
                    continue

                if not ffmpeg_ready:
                    skipped.append(f"{item.name} (non-WAV source and ffmpeg not found)")
                    continue

                src_suffix = Path(urlparse(item.url).path).suffix or ".bin"
                with tempfile.NamedTemporaryFile(delete=False, suffix=src_suffix) as tmp:
                    src_path = Path(tmp.name)
                try:
                    _download_to_path_streaming(item.url, src_path, progress_cb=progress_cb)
                except Exception as exc:
                    src_path.unlink(missing_ok=True)
                    skipped.append(f"{item.name} ({exc})")
                    continue

                try:
                    if _convert_to_wav_ffmpeg(src_path, out_path):
                        imported.append(str(out_path))
                        existing_basenames.add(base_key)
                        batch_basenames.add(base_key)
                    else:
                        skipped.append(f"{item.name} (ffmpeg conversion failed)")
                finally:
                    src_path.unlink(missing_ok=True)

            self.finished.emit({"imported": imported, "skipped": skipped})
        except Exception as exc:
            self.error.emit(str(exc))


class FileImportWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(object)

    def __init__(self, source_files: list[str], sounds_dir: Path) -> None:
        super().__init__()
        self.source_files = [str(p) for p in source_files]
        self.sounds_dir = Path(sounds_dir)

    def run(self) -> None:
        try:
            self.sounds_dir.mkdir(parents=True, exist_ok=True)
            imported: list[str] = []
            skipped: list[str] = []
            existing_basenames = {p.stem.lower() for p in self.sounds_dir.iterdir() if p.is_file()}
            total_items = max(1, len(self.source_files))

            for idx, src_str in enumerate(self.source_files, start=1):
                src = Path(src_str)
                self.progress.emit(
                    {
                        "phase": "item-start",
                        "item_name": src.name,
                        "item_index": idx,
                        "total_items": total_items,
                    }
                )
                if not src.exists():
                    skipped.append(f"{src.name} (missing)")
                    continue
                base = _safe_name(src.stem)
                if base.lower() in existing_basenames:
                    skipped.append(f"{src.name} (duplicate name)")
                    continue

                dst = self.sounds_dir / f"{base}{src.suffix.lower()}"
                total_bytes = 0
                try:
                    total_bytes = max(0, int(src.stat().st_size))
                except Exception:
                    total_bytes = 0

                copied = 0
                try:
                    with src.open("rb") as in_fh, dst.open("wb") as out_fh:
                        while True:
                            chunk = in_fh.read(1024 * 1024)
                            if not chunk:
                                break
                            out_fh.write(chunk)
                            copied += len(chunk)
                            self.progress.emit(
                                {
                                    "phase": "copy",
                                    "item_name": src.name,
                                    "item_index": idx,
                                    "total_items": total_items,
                                    "downloaded_bytes": int(copied),
                                    "total_bytes": int(total_bytes) if total_bytes > 0 else None,
                                }
                            )
                    try:
                        shutil.copystat(src, dst)
                    except Exception:
                        pass
                    imported.append(str(dst))
                    existing_basenames.add(base.lower())
                except Exception as exc:
                    try:
                        dst.unlink(missing_ok=True)
                    except Exception:
                        pass
                    skipped.append(f"{src.name} ({exc})")

            self.finished.emit({"imported": imported, "skipped": skipped})
        except Exception as exc:
            self.error.emit(str(exc))


class YoutubeImportWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, url: str, sounds_dir: Path, clip_seconds: float = 5.0) -> None:
        super().__init__()
        self.url = url
        self.sounds_dir = sounds_dir
        self.clip_seconds = max(1.0, float(clip_seconds))

    def run(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="sbz_ytdl_"))
        try:
            try:
                from yt_dlp import YoutubeDL  # type: ignore
            except Exception:
                raise RuntimeError("yt-dlp is not installed. Install with: pip install yt-dlp")

            ffmpeg_exe = _get_ffmpeg_exe()
            if ffmpeg_exe is None:
                raise RuntimeError("ffmpeg not found. Required for YouTube import.")

            self.sounds_dir.mkdir(parents=True, exist_ok=True)
            outtmpl = str(temp_dir / "%(title).80s.%(ext)s")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": outtmpl,
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
                title = info.get("title") or "youtube_clip"

            downloaded = [p for p in temp_dir.iterdir() if p.is_file()]
            if not downloaded:
                raise RuntimeError("Failed to download YouTube audio.")
            src = max(downloaded, key=lambda p: p.stat().st_mtime)

            base = _safe_name(str(title))
            out_path = _unique_path(self.sounds_dir / f"{base}.mp3")
            cmd = [
                ffmpeg_exe,
                "-y",
                "-v",
                "error",
                "-ss",
                "0",
                "-t",
                f"{self.clip_seconds:.3f}",
                "-i",
                str(src),
                "-vn",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-b:a",
                "192k",
                str(out_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not out_path.exists():
                raise RuntimeError("Failed to convert YouTube clip.")

            self.finished.emit({"imported": [str(out_path)], "skipped": []})
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class PreviewWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, item: RemoteSoundItem, volume: float = 0.5) -> None:
        super().__init__()
        self.item = item
        self.volume = max(0.0, min(1.0, float(volume)))

    def run(self) -> None:
        temp_src: Path | None = None
        try:
            src_suffix = Path(urlparse(self.item.url).path).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=src_suffix) as tmp:
                temp_src = Path(tmp.name)
            _download_to_path_streaming(self.item.url, temp_src)

            audio, sr = _decode_audio_for_preview(temp_src)
            np.clip(audio, -1.0, 1.0, out=audio)
            self.finished.emit(
                {"item": self.item, "audio": np.ascontiguousarray(audio, dtype=np.float32), "sr": sr}
            )
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if temp_src is not None:
                temp_src.unlink(missing_ok=True)


class SoundLoadWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(object)

    def __init__(self, engine: AudioEngine, imported_paths: list[str]) -> None:
        super().__init__()
        self.engine = engine
        self.imported_paths = [str(p) for p in imported_paths]

    def run(self) -> None:
        try:
            loaded = 0
            failed: list[str] = []
            total_items = max(1, len(self.imported_paths))
            for idx, path_str in enumerate(self.imported_paths, start=1):
                path = Path(path_str)
                sound_name = path.stem
                self.progress.emit(
                    {
                        "phase": "cache",
                        "item_name": path.name,
                        "item_index": idx,
                        "total_items": total_items,
                    }
                )
                try:
                    self.engine.soundboard.load_audio_file(sound_name, path)
                    loaded += 1
                except Exception as exc:
                    failed.append(f"{path.name}: {exc}")
            self.finished.emit({"loaded": loaded, "failed": failed})
        except Exception as exc:
            self.error.emit(str(exc))


class AudioDecodeWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, source_path: str | Path) -> None:
        super().__init__()
        self.source_path = Path(source_path)

    def run(self) -> None:
        try:
            audio, sr = _decode_audio_for_preview(self.source_path)
            self.finished.emit(
                {
                    "path": str(self.source_path),
                    "audio": np.ascontiguousarray(audio, dtype=np.float32),
                    "sr": int(sr),
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))


class TrimApplyWorker(QObject):
    """Runs ffmpeg trim + sound reload off the UI thread."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        ffmpeg_exe: str,
        src: Path,
        start: float,
        length: float,
        engine: AudioEngine,
    ) -> None:
        super().__init__()
        self.ffmpeg_exe = ffmpeg_exe
        self.src = Path(src)
        self.start = float(start)
        self.length = float(length)
        self.engine = engine

    def run(self) -> None:
        trimmed_tmp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=self.src.suffix or ".wav") as tmp:
                trimmed_tmp = Path(tmp.name)
            cmd = [
                self.ffmpeg_exe,
                "-y",
                "-v",
                "error",
                "-ss",
                f"{self.start:.3f}",
                "-t",
                f"{self.length:.3f}",
                "-i",
                str(self.src),
                "-vn",
                "-ar",
                "48000",
                "-ac",
                "2",
                str(trimmed_tmp),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not trimmed_tmp.exists():
                trimmed_tmp.unlink(missing_ok=True)
                self.error.emit("ffmpeg failed to trim the audio.")
                return

            trimmed_tmp.replace(self.src)

            try:
                self.engine.soundboard.load_audio_file(self.src.stem, self.src)
            except Exception as exc:
                self.finished.emit({"ok": True, "reload_error": str(exc)})
                return
            self.finished.emit({"ok": True})
        except Exception as exc:
            if trimmed_tmp is not None:
                trimmed_tmp.unlink(missing_ok=True)
            self.error.emit(str(exc))


class RouteApplyWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        engine: AudioEngine,
        restart_fn,
        input_device,
        output_device,
        old_input_device,
        old_output_device,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.restart_fn = restart_fn
        self.input_device = input_device
        self.output_device = output_device
        self.old_input_device = old_input_device
        self.old_output_device = old_output_device

    def run(self) -> None:
        try:
            self.engine.input_device = self.input_device
            self.engine.output_device = self.output_device
            ok, err = self.restart_fn()
            if ok:
                self.finished.emit({"ok": True})
                return

            self.engine.input_device = self.old_input_device
            self.engine.output_device = self.old_output_device
            rollback_ok, rollback_err = self.restart_fn()
            self.finished.emit(
                {
                    "ok": False,
                    "err": err,
                    "rollback_ok": rollback_ok,
                    "rollback_err": rollback_err,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))


class StartupUpdateWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    stage = pyqtSignal(str)

    def __init__(self, update: UpdateInfo, update_root: Path) -> None:
        super().__init__()
        self.update = update
        self.update_root = Path(update_root)

    def run(self) -> None:
        try:
            self.update_root.mkdir(parents=True, exist_ok=True)

            def on_progress(downloaded: int, total: int | None) -> None:
                total_value = int(total) if isinstance(total, int) and total > 0 else -1
                self.progress.emit(int(max(0, downloaded)), total_value)

            if self.update.is_delta:
                # Delta: download individual patch files
                self.stage.emit("download")
                delta_dir = self.update_root / "delta_files"
                delta_pairs = [(df.relative_path, df.url) for df in self.update.delta_files]
                download_delta_files(delta_pairs, delta_dir, progress_cb=on_progress)
                self.finished.emit(
                    {
                        "mode": "delta",
                        "temp_dir": str(delta_dir),
                        "version": str(self.update.version),
                    }
                )
            else:
                # Full: download zip package
                self.stage.emit("download")
                package_path = self.update_root / "update.pkg"
                download_file(self.update.full_url, package_path, progress_cb=on_progress)
                # Keep the .pkg inside a dedicated temp dir for the applier
                full_dir = self.update_root / "full_pkg"
                full_dir.mkdir(parents=True, exist_ok=True)
                final_pkg = full_dir / "update.pkg"
                package_path.replace(final_pkg)
                self.finished.emit(
                    {
                        "mode": "full",
                        "temp_dir": str(full_dir),
                        "version": str(self.update.version),
                    }
                )
        except Exception as exc:
            self.error.emit(str(exc))


class SmoothListWidget(QListWidget):
    def __init__(self, slow_factor: float = 0.6, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._slow_factor = max(0.2, min(1.0, float(slow_factor)))
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.verticalScrollBar().setSingleStep(18)
        self._target_value = self.verticalScrollBar().value()
        self._scroll_anim = QPropertyAnimation(self.verticalScrollBar(), b"value", self)
        self._scroll_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._scroll_anim.setDuration(110)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        bar = self.verticalScrollBar()
        pixel_delta = event.pixelDelta().y()
        angle_delta = event.angleDelta().y()
        if pixel_delta == 0 and angle_delta == 0:
            super().wheelEvent(event)
            return

        if self._scroll_anim.state() != QPropertyAnimation.State.Running:
            self._target_value = bar.value()

        if pixel_delta != 0:
            distance = int(pixel_delta * self._slow_factor)
        else:
            distance = int((angle_delta / 120.0) * bar.singleStep() * 2.6 * self._slow_factor)

        target = self._target_value - distance
        target = max(bar.minimum(), min(bar.maximum(), target))
        self._target_value = target
        if target == bar.value():
            event.accept()
            return

        delta_abs = abs(target - bar.value())
        duration = max(70, min(180, int(delta_abs * 0.45)))
        self._scroll_anim.stop()
        self._scroll_anim.setDuration(duration)
        self._scroll_anim.setStartValue(bar.value())
        self._scroll_anim.setEndValue(target)
        self._scroll_anim.start()
        event.accept()


def _scroll_to_parent_area(widget: QWidget, event) -> None:
    delta = event.pixelDelta().y()
    if delta == 0:
        angle_delta = event.angleDelta().y()
        if angle_delta != 0:
            delta = int((angle_delta / 120.0) * 36)
    if delta == 0:
        return
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea):
            bar = parent.verticalScrollBar()
            if bar is not None and bar.maximum() > bar.minimum():
                bar.setValue(bar.value() - delta)
                return
        parent = parent.parentWidget()


class NoWheelSlider(QSlider):
    def wheelEvent(self, event) -> None:  # type: ignore[override]
        _scroll_to_parent_area(self, event)
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:  # type: ignore[override]
        _scroll_to_parent_area(self, event)
        event.ignore()


def _apply_dwm_rounded_corners(widget: QWidget) -> None:
    """Ask the Windows 11+ DWM compositor to clip the window to round corners.

    This is the most reliable way to eliminate black/sharp ghost borders
    because the compositor itself applies the clipping before the surface
    reaches the screen — no `QRegion` mask or QPainter trick can match it.
    On older Windows versions or non-Windows platforms the call silently
    does nothing.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import ctypes.wintypes

        hwnd = int(widget.winId())
        DWMWA_WINDOW_CORNER_PREFERENCE = 33  # Windows 11 Build 22000+
        DWMWCP_ROUND = 2  # fully rounded
        preference = ctypes.c_int(DWMWCP_ROUND)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            ctypes.wintypes.HWND(hwnd),
            DWMWA_WINDOW_CORNER_PREFERENCE,
            ctypes.byref(preference),
            ctypes.sizeof(preference),
        )
        # Suppress the DWM-drawn window border (appears as a white/light
        # outline on Windows 11).  DWMWA_COLOR_NONE = 0xFFFFFFFE.
        DWMWA_BORDER_COLOR = 34
        DWMWA_COLOR_NONE = 0xFFFFFFFE
        border_color = ctypes.wintypes.DWORD(DWMWA_COLOR_NONE)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            ctypes.wintypes.HWND(hwnd),
            DWMWA_BORDER_COLOR,
            ctypes.byref(border_color),
            ctypes.sizeof(border_color),
        )
    except Exception:
        pass  # Pre-Win11 or missing dwmapi – fine, CSS handles it.


class RoundedContainer(QFrame):
    """Main visual container with CSS-driven rounded corners.

    Because the top-level window uses ``WA_TranslucentBackground``, the
    compositor blends only the painted regions. CSS ``border-radius`` on
    this ``QFrame`` (with ``WA_StyledBackground``) correctly leaves the
    corners transparent, giving perfectly smooth rounded edges.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("RoundedContainer")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.NoFrame)


class _WindowBackdrop(QWidget):
    """Central widget of the main window.

    Fully transparent backdrop – all visuals come from the child
    ``RoundedContainer`` via CSS.  No manual shadow painting so that
    nothing leaks outside the rounded container's bounds.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WindowBackdrop")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


class ButtonMotionFilter(QObject):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._effects: dict[QPushButton, QGraphicsDropShadowEffect] = {}
        self._animations: dict[QPushButton, QPropertyAnimation] = {}
        self._glow_colors: dict[QPushButton, QColor] = {}

    def attach(self, button: QPushButton, glow: str = "#67e8f9") -> None:
        if button in self._effects:
            return
        effect = QGraphicsDropShadowEffect(button)
        effect.setBlurRadius(12.0)
        effect.setOffset(0.0, 2.0)
        effect.setColor(QColor(7, 14, 31, 155))
        button.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"blurRadius", button)
        anim.setDuration(150)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._effects[button] = effect
        self._animations[button] = anim
        self._glow_colors[button] = QColor(glow)
        button.installEventFilter(self)

    def _animate(self, button: QPushButton, blur: float, y_offset: float, color: QColor) -> None:
        effect = self._effects.get(button)
        anim = self._animations.get(button)
        if effect is None or anim is None:
            return
        anim.stop()
        anim.setStartValue(float(effect.blurRadius()))
        anim.setEndValue(float(blur))
        anim.start()
        effect.setOffset(0.0, float(y_offset))
        effect.setColor(color)

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if not isinstance(obj, QPushButton) or obj not in self._effects:
            return super().eventFilter(obj, event)

        glow = self._glow_colors.get(obj, QColor("#67e8f9"))
        typ = event.type()
        if typ == QEvent.Type.Enter:
            self._animate(obj, blur=24.0, y_offset=0.0, color=QColor(glow.red(), glow.green(), glow.blue(), 150))
        elif typ == QEvent.Type.Leave:
            self._animate(obj, blur=12.0, y_offset=2.0, color=QColor(7, 14, 31, 155))
        elif typ == QEvent.Type.MouseButtonPress:
            self._animate(obj, blur=9.0, y_offset=1.0, color=QColor(glow.red(), glow.green(), glow.blue(), 110))
        elif typ == QEvent.Type.MouseButtonRelease:
            if obj.underMouse():
                self._animate(obj, blur=24.0, y_offset=0.0, color=QColor(glow.red(), glow.green(), glow.blue(), 150))
            else:
                self._animate(obj, blur=12.0, y_offset=2.0, color=QColor(7, 14, 31, 155))
        return super().eventFilter(obj, event)


class MacTrafficButton(QPushButton):
    def __init__(self, role: str, parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        self._role = "close" if str(role).lower() == "close" else "minimize"
        self._hovered = False
        self._scale_factor = 1.0
        self._scale_anim = QPropertyAnimation(self, b"scaleFactor", self)
        self._scale_anim.setDuration(160)
        self._scale_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.setFixedSize(16, 16)
        self.setFlat(True)
        self.setCheckable(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    @pyqtProperty(float)
    def scaleFactor(self) -> float:
        return float(self._scale_factor)

    @scaleFactor.setter
    def scaleFactor(self, value: float) -> None:
        self._scale_factor = float(max(0.9, min(1.2, value)))
        self.update()

    def _animate_scale(self, target: float) -> None:
        self._scale_anim.stop()
        self._scale_anim.setStartValue(float(self._scale_factor))
        self._scale_anim.setEndValue(float(target))
        self._scale_anim.start()

    def enterEvent(self, event) -> None:  # type: ignore[override]
        self._hovered = True
        self._animate_scale(1.05)
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hovered = False
        self._animate_scale(1.0)
        self.update()
        super().leaveEvent(event)

    def _base_color(self) -> QColor:
        if self._role == "close":
            return QColor("#FF5F57")
        return QColor("#F4BF4F")

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(0.8, 0.8, -0.8, -0.8)
        if self._scale_factor != 1.0:
            center = rect.center()
            width = rect.width() * self._scale_factor
            height = rect.height() * self._scale_factor
            rect = QRectF(center.x() - (width * 0.5), center.y() - (height * 0.5), width, height)

        side = min(rect.width(), rect.height())
        rect = QRectF(
            rect.center().x() - (side * 0.5),
            rect.center().y() - (side * 0.5),
            side,
            side,
        )
        path = QPainterPath()
        path.addEllipse(rect)

        base = self._base_color()
        if self._hovered:
            base = base.lighter(106)

        grad = QLinearGradient(rect.left(), rect.top(), rect.left(), rect.bottom())
        grad.setColorAt(0.0, base.lighter(120))
        grad.setColorAt(0.52, base)
        grad.setColorAt(1.0, base.darker(122))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(grad)
        painter.drawPath(path)

        border_pen = QPen(QColor(14, 22, 35, 72), 0.9)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        top_highlight_pen = QPen(QColor(255, 255, 255, 58), 0.9)
        painter.setPen(top_highlight_pen)
        painter.drawLine(
            QPointF(rect.left() + 3.0, rect.top() + 2.0),
            QPointF(rect.right() - 3.0, rect.top() + 2.0),
        )

        inner_shadow_pen = QPen(QColor(0, 0, 0, 54), 1.0)
        painter.setPen(inner_shadow_pen)
        painter.drawLine(
            QPointF(rect.left() + 3.1, rect.bottom() - 1.9),
            QPointF(rect.right() - 3.1, rect.bottom() - 1.9),
        )

        if self._hovered:
            icon_pen = QPen(QColor(24, 32, 44, 155), 1.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(icon_pen)
            mid_x = rect.center().x()
            mid_y = rect.center().y()
            if self._role == "minimize":
                painter.drawLine(QPointF(mid_x - 2.6, mid_y + 0.2), QPointF(mid_x + 2.6, mid_y + 0.2))
            else:
                painter.drawLine(QPointF(mid_x - 2.0, mid_y - 2.0), QPointF(mid_x + 2.0, mid_y + 2.0))
                painter.drawLine(QPointF(mid_x + 2.0, mid_y - 2.0), QPointF(mid_x - 2.0, mid_y + 2.0))


class LocalTileDelegate(QStyledItemDelegate):
    _cached_tile_font: QFont | None = None

    @staticmethod
    def _toggle_check_state(model, index) -> None:
        state = index.data(int(Qt.ItemDataRole.CheckStateRole))
        checked = state == Qt.CheckState.Checked
        next_state = Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked
        model.setData(index, next_state, int(Qt.ItemDataRole.CheckStateRole))

    def editorEvent(self, event, model, option, index) -> bool:  # type: ignore[override]
        if not (index.flags() & Qt.ItemFlag.ItemIsUserCheckable):
            return super().editorEvent(event, model, option, index)

        typ = event.type()
        if typ in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
            button = getattr(event, "button", lambda: Qt.MouseButton.NoButton)()
            if button != Qt.MouseButton.LeftButton:
                return False
            pos = getattr(event, "position", None)
            if pos is None:
                return False
            point = pos().toPoint()
            tile_rect = option.rect.adjusted(8, 8, -8, -8)
            if not tile_rect.contains(point):
                return False
            if typ == QEvent.Type.MouseButtonRelease:
                self._toggle_check_state(model, index)
            return True

        if typ == QEvent.Type.KeyPress:
            key = getattr(event, "key", lambda: 0)()
            if key in (int(Qt.Key.Key_Space), int(Qt.Key.Key_Return), int(Qt.Key.Key_Enter)):
                self._toggle_check_state(model, index)
                return True

        return super().editorEvent(event, model, option, index)

    def paint(self, painter: QPainter, option, index) -> None:  # type: ignore[override]
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = option.rect.adjusted(10, 8, -10, -8)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        checked = index.data(int(Qt.ItemDataRole.CheckStateRole)) == Qt.CheckState.Checked
        if hovered:
            rect.translate(0, -2)

        base_color = QColor(60, 78, 102, 180)
        if hovered:
            base_color = QColor(80, 106, 146, 196)
        if selected:
            base_color = QColor(96, 138, 190, 215)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(8, 16, 28, 78))
        painter.drawRoundedRect(rect.adjusted(0, 2, 0, 2), 14, 14)

        border_color = QColor(180, 200, 230, 62)
        border_width = 1
        if hovered and not selected:
            border_color = QColor(180, 220, 255, 120)
        if selected:
            border_color = QColor(190, 235, 255, 185)

        painter.setBrush(base_color)
        pen = QPen(border_color, border_width)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 14, 14)

        text = str(index.data(int(Qt.ItemDataRole.DisplayRole)) or "")
        text_rect = rect.adjusted(10, 0, -10, 0)
        if index.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            text_rect.adjust(0, 0, -18, 0)
        if LocalTileDelegate._cached_tile_font is None:
            f = QFont(option.font)
            f.setPointSize(9)
            f.setWeight(QFont.Weight.Medium)
            LocalTileDelegate._cached_tile_font = f
        fm = QFontMetrics(LocalTileDelegate._cached_tile_font)
        text = fm.elidedText(text, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.setFont(LocalTileDelegate._cached_tile_font)
        painter.setPen(QColor("#e2e8f0"))
        painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)

        if index.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            check_rect = QRect(rect.right() - 20, rect.top() + 6, 12, 12)
            painter.setBrush(QColor("#38bdf8") if checked else QColor("#64748b"))
            painter.setPen(QPen(QColor("#cbd5e1"), 1))
            painter.drawEllipse(check_rect)
            if checked:
                painter.setPen(QPen(QColor("#0f172a"), 2))
                painter.drawLine(check_rect.left() + 3, check_rect.center().y(), check_rect.left() + 5, check_rect.bottom() - 3)
                painter.drawLine(check_rect.left() + 5, check_rect.bottom() - 3, check_rect.right() - 2, check_rect.top() + 3)

        painter.restore()


class TrimTimelineWidget(QWidget):
    startChanged = pyqtSignal(int)
    endChanged = pyqtSignal(int)
    playheadChanged = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._duration_ms = 1
        self._start_ms = 0
        self._end_ms = 1
        self._playhead_ms = 0
        self._drag_target: str | None = None
        self.setMinimumHeight(118)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def duration_ms(self) -> int:
        return self._duration_ms

    def start_ms(self) -> int:
        return self._start_ms

    def end_ms(self) -> int:
        return self._end_ms

    def playhead_ms(self) -> int:
        return self._playhead_ms

    def set_duration_ms(self, value: int) -> None:
        self._duration_ms = max(1, int(value))
        self._start_ms = max(0, min(self._start_ms, max(0, self._duration_ms - 1)))
        self._end_ms = max(self._start_ms + 1, min(self._end_ms, self._duration_ms))
        self._playhead_ms = max(self._start_ms, min(self._playhead_ms, self._end_ms))
        self.update()

    def set_range_ms(self, start_ms: int, end_ms: int, emit: bool = False) -> None:
        start = max(0, min(int(start_ms), self._duration_ms - 1))
        end = max(start + 1, min(int(end_ms), self._duration_ms))
        changed_start = start != self._start_ms
        changed_end = end != self._end_ms
        self._start_ms = start
        self._end_ms = end
        if self._playhead_ms < self._start_ms:
            self._playhead_ms = self._start_ms
            if emit:
                self.playheadChanged.emit(self._playhead_ms)
        elif self._playhead_ms > self._end_ms:
            self._playhead_ms = self._end_ms
            if emit:
                self.playheadChanged.emit(self._playhead_ms)
        if emit and changed_start:
            self.startChanged.emit(self._start_ms)
        if emit and changed_end:
            self.endChanged.emit(self._end_ms)
        self.update()

    def set_playhead_ms(self, playhead_ms: int, emit: bool = False) -> None:
        value = max(self._start_ms, min(int(playhead_ms), self._end_ms))
        if value == self._playhead_ms:
            return
        self._playhead_ms = value
        if emit:
            self.playheadChanged.emit(self._playhead_ms)
        self.update()

    def _track_rect(self) -> QRectF:
        r = self.rect().adjusted(14, 24, -14, -14)
        if r.width() < 20:
            return QRectF(0.0, 0.0, 1.0, 1.0)
        return QRectF(float(r.left()), float(r.top()), float(r.width()), float(r.height()))

    def _ms_to_x(self, ms: int) -> float:
        track = self._track_rect()
        ratio = max(0.0, min(1.0, float(ms) / float(max(1, self._duration_ms))))
        return track.left() + (ratio * track.width())

    def _x_to_ms(self, x: float) -> int:
        track = self._track_rect()
        if track.width() <= 0.0:
            return 0
        ratio = (x - track.left()) / track.width()
        ratio = max(0.0, min(1.0, ratio))
        return int(round(ratio * self._duration_ms))

    @staticmethod
    def _fmt_ms(ms: int) -> str:
        s = max(0, int(ms)) / 1000.0
        minutes = int(s // 60)
        seconds = s - (minutes * 60)
        return f"{minutes:02d}:{seconds:05.2f}"

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        outer = self.rect().adjusted(1, 1, -1, -1)
        outer_f = QRectF(float(outer.left()), float(outer.top()), float(outer.width()), float(outer.height()))
        painter.setBrush(QColor("#1e293b"))
        painter.setPen(QPen(QColor("#334155"), 1))
        painter.drawRoundedRect(outer_f, 14, 14)

        track = self._track_rect()
        painter.setBrush(QColor("#0f172a"))
        painter.setPen(QPen(QColor("#334155"), 1))
        painter.drawRoundedRect(track, 10, 10)

        bar_count = max(32, int(track.width() / 7))
        center_y = track.center().y()
        for i in range(bar_count):
            t = i / max(1, bar_count - 1)
            x = track.left() + (t * track.width())
            wave = 0.15 + (abs(math.sin((t * 9.6) + 0.6)) * 0.85)
            amp = (track.height() * wave) * 0.5
            alpha = 45 + int(55 * wave)
            painter.setPen(QPen(QColor(71, 85, 105, alpha), 2))
            painter.drawLine(QPointF(x, center_y - amp), QPointF(x, center_y + amp))

        start_x = self._ms_to_x(self._start_ms)
        end_x = self._ms_to_x(self._end_ms)
        play_x = self._ms_to_x(self._playhead_ms)

        selection = QRectF(start_x, track.top(), max(2.0, end_x - start_x), track.height())
        painter.setBrush(QColor(56, 189, 248, 48))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(selection, 8, 8)

        def draw_pin(x_pos: float, color: QColor) -> None:
            pin_top = track.top() - 11.0
            pin_bottom = track.bottom() + 8.0
            painter.setPen(QPen(color, 2.2))
            painter.drawLine(QPointF(x_pos, pin_top + 4.0), QPointF(x_pos, pin_bottom))
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor("#cbd5e1"), 1))
            painter.drawEllipse(QPointF(x_pos, pin_top), 5.5, 5.5)

        draw_pin(start_x, QColor("#38bdf8"))
        draw_pin(end_x, QColor("#7dd3fc"))
        painter.setPen(QPen(QColor("#e2e8f0"), 1.8))
        painter.drawLine(QPointF(play_x, track.top() - 7.0), QPointF(play_x, track.bottom() + 7.0))
        painter.setBrush(QBrush(QColor("#e2e8f0")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(play_x, track.top() - 7.0), 4.0, 4.0)

        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(
            QRectF(track.left(), outer_f.top() + 2.0, track.width(), 16.0),
            int(Qt.AlignmentFlag.AlignCenter),
            f"{self._fmt_ms(self._start_ms)}  -  {self._fmt_ms(self._end_ms)}",
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        pos = event.position()
        track = self._track_rect()
        x = float(pos.x())
        y = float(pos.y())

        start_x = self._ms_to_x(self._start_ms)
        end_x = self._ms_to_x(self._end_ms)
        play_x = self._ms_to_x(self._playhead_ms)
        pin_top = track.top() - 20.0
        pin_bottom = track.bottom() + 20.0
        in_pin_band = pin_top <= y <= pin_bottom

        if in_pin_band and abs(x - start_x) <= 15.0:
            self._drag_target = "start"
        elif in_pin_band and abs(x - end_x) <= 15.0:
            self._drag_target = "end"
        elif in_pin_band and abs(x - play_x) <= 14.0:
            self._drag_target = "playhead"
        elif track.contains(pos):
            self._drag_target = "playhead"
        else:
            super().mousePressEvent(event)
            return

        self._apply_drag_x(x)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        x = float(event.position().x())
        y = float(event.position().y())
        if self._drag_target is not None:
            self._apply_drag_x(x)
            event.accept()
            return

        track = self._track_rect()
        start_x = self._ms_to_x(self._start_ms)
        end_x = self._ms_to_x(self._end_ms)
        play_x = self._ms_to_x(self._playhead_ms)
        pin_top = track.top() - 20.0
        pin_bottom = track.bottom() + 20.0
        close_to_pin = pin_top <= y <= pin_bottom and min(abs(x - start_x), abs(x - end_x), abs(x - play_x)) <= 14.0
        if close_to_pin:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._drag_target = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def _apply_drag_x(self, x: float) -> None:
        ms = self._x_to_ms(x)
        if self._drag_target == "start":
            start = min(ms, self._end_ms - 1)
            if start != self._start_ms:
                self._start_ms = start
                self.startChanged.emit(self._start_ms)
            if self._playhead_ms < self._start_ms:
                self._playhead_ms = self._start_ms
                self.playheadChanged.emit(self._playhead_ms)
        elif self._drag_target == "end":
            end = max(ms, self._start_ms + 1)
            if end != self._end_ms:
                self._end_ms = end
                self.endChanged.emit(self._end_ms)
            if self._playhead_ms > self._end_ms:
                self._playhead_ms = self._end_ms
                self.playheadChanged.emit(self._playhead_ms)
        else:
            playhead = max(self._start_ms, min(ms, self._end_ms))
            if playhead != self._playhead_ms:
                self._playhead_ms = playhead
                self.playheadChanged.emit(self._playhead_ms)
        self.update()


class TrimEditorDialog(QDialog):
    closed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TrimEditorDialog")
        self.setWindowTitle("Trim Editor")
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.resize(640, 300)
        self.setMinimumSize(560, 270)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 12)
        root.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(10)
        self.trim_target_label = QLabel("No sound selected")
        self.trim_target_label.setObjectName("TrimTarget")
        self.trim_time_label = QLabel("00:00.00 / 00:00.00")
        self.trim_time_label.setObjectName("TrimTime")
        header.addWidget(self.trim_target_label, 1)
        header.addWidget(self.trim_time_label, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        root.addLayout(header)

        self.timeline = TrimTimelineWidget()
        root.addWidget(self.timeline)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.trim_play_pause_btn = QPushButton("Play")
        self.trim_play_pause_btn.setProperty("variant", "cyan")
        self.trim_stop_btn = QPushButton("Stop")
        self.trim_stop_btn.setProperty("variant", "amber")
        self.apply_trim_btn = QPushButton("Apply Trim")
        self.apply_trim_btn.setProperty("variant", "success")
        self.close_trim_editor_btn = QPushButton("Close")
        self.close_trim_editor_btn.setProperty("variant", "secondary")
        controls.addWidget(self.trim_play_pause_btn)
        controls.addWidget(self.trim_stop_btn)
        controls.addWidget(self.apply_trim_btn)
        controls.addStretch(1)
        controls.addWidget(self.close_trim_editor_btn)
        root.addLayout(controls)

        self.setStyleSheet(
            """
            QDialog#TrimEditorDialog {
                background: rgba(22, 33, 50, 242);
                border: 1px solid rgba(148, 163, 184, 42);
                border-radius: 20px;
            }
            QDialog#TrimEditorDialog QLabel#TrimTarget {
                font-size: 15px;
                font-weight: 600;
                color: #edf3fb;
            }
            QDialog#TrimEditorDialog QLabel#TrimTime {
                font-size: 13px;
                font-weight: 600;
                color: #bfd0e4;
                padding-right: 2px;
            }
            """
        )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.closed.emit()
        super().closeEvent(event)


class FramelessImporterDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint
        )
        # Enable true alpha transparency – the ImporterShell's CSS
        # border-radius handles the visual rounding against a transparent
        # dialog background with no jagged polygon mask artifacts.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._corner_radius = 28.0
        self._dragging = False
        self._drag_offset = QPoint()

    def _apply_window_mask(self) -> None:
        # With WA_TranslucentBackground the compositor handles transparency,
        # so a pixel-aligned QRegion mask is no longer needed (it caused
        # jagged edges and ghost-rectangle artifacts).
        self.clearMask()

    @staticmethod
    def _is_drag_exempt_widget(widget: QWidget | None) -> bool:
        current = widget
        while current is not None:
            if isinstance(
                current,
                (QPushButton, QLineEdit, QComboBox, QSlider, QListWidget, QAbstractItemView, QScrollArea),
            ):
                return True
            current = current.parentWidget()
        return False

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        child = self.childAt(event.position().toPoint())
        if self._is_drag_exempt_widget(child):
            super().mousePressEvent(event)
            return
        self._dragging = True
        self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._dragging = False
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_window_mask()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._apply_window_mask()
        _apply_dwm_rounded_corners(self)


class UpdateDialog(QDialog):
    updateRequested = pyqtSignal()

    def __init__(
        self,
        current_version: str,
        new_version: str,
        parent: QWidget | None = None,
        mandatory: bool = False,
    ) -> None:
        super().__init__(parent)
        self._busy = False
        self._mandatory = bool(mandatory)
        self._allow_close = not self._mandatory
        self.setObjectName("UpdateDialog")
        self.setWindowTitle("Update Available")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        if self._mandatory:
            self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QLabel("Update Available")
        title.setObjectName("UpdateTitle")
        subtitle = QLabel(
            f"A newer version is available.\nCurrent: {current_version}\nNew: {new_version}"
        )
        subtitle.setObjectName("UpdateSubtitle")

        self.status_label = QLabel("Ready to update.")
        self.status_label.setObjectName("UpdateStatus")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(10)
        button_row.addStretch(1)
        self.skip_button = QPushButton("Skip")
        self.skip_button.setProperty("variant", "secondary")
        self.update_button = QPushButton("Update")
        self.update_button.setProperty("variant", "primary")
        button_row.addWidget(self.skip_button)
        button_row.addWidget(self.update_button)

        root.addWidget(title)
        root.addWidget(subtitle)
        root.addWidget(self.status_label)
        root.addWidget(self.progress_bar)
        root.addLayout(button_row)

        self.skip_button.clicked.connect(self.reject)
        self.update_button.clicked.connect(self.updateRequested.emit)
        if self._mandatory:
            self.skip_button.setEnabled(False)

        self.setStyleSheet(
            """
            QDialog#UpdateDialog {
                background: rgba(22, 33, 50, 245);
                border: 1px solid rgba(148, 163, 184, 44);
                border-radius: 14px;
            }
            QLabel#UpdateTitle {
                color: #edf3fb;
                font-size: 18px;
                font-weight: 600;
            }
            QLabel#UpdateSubtitle {
                color: #c5d4e8;
                font-size: 13px;
                line-height: 1.35;
            }
            QLabel#UpdateStatus {
                color: #dbe8f7;
                font-size: 12px;
                font-weight: 500;
            }
            QProgressBar {
                background: rgba(11, 21, 36, 220);
                border: 1px solid rgba(148, 163, 184, 44);
                border-radius: 8px;
                min-height: 18px;
                color: #eaf2fb;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #59d7ff;
                border-radius: 7px;
            }
            """
        )

    def begin_update(self) -> None:
        self._busy = True
        self.update_button.setText("Update")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Downloading update...")
        self.update_button.setEnabled(False)
        self.skip_button.setEnabled(False)

    def set_download_progress(self, downloaded: int, total: int) -> None:
        self.progress_bar.setVisible(True)
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(max(0, min(total, downloaded)))
            self.status_label.setText(
                f"Downloading update... {downloaded / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB"
            )
        else:
            self.progress_bar.setRange(0, 0)
            self.status_label.setText("Downloading update...")

    def set_extracting(self) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Extracting update package...")

    def set_installing(self) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Installing update and relaunching...")

    def finish_success(self) -> None:
        self._busy = False
        self._allow_close = True

    def set_error(self, message: str) -> None:
        self._busy = False
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Update failed: {message}")
        self.update_button.setText("Retry")
        self.update_button.setEnabled(True)
        if self._mandatory:
            self.skip_button.setEnabled(False)
        else:
            self.skip_button.setEnabled(True)

    def reject(self) -> None:  # type: ignore[override]
        if self._mandatory and not self._allow_close:
            return
        super().reject()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._busy or (self._mandatory and not self._allow_close):
            event.ignore()
            return
        super().closeEvent(event)


class SoundboardWindow(QMainWindow):
    DEFAULT_KEYS = list("1234567890qwertyuiopasdfghjklzxcvbnm")
    DEFAULT_MIC_VOLUME = 60
    DEFAULT_SOUNDBOARD_VOLUME = 15

    def __init__(self, launched_from_startup: bool = False) -> None:
        super().__init__()
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        # Enable true alpha transparency so the compositor renders only the
        # painted regions. The RoundedContainer's CSS border-radius produces
        # smooth rounded corners against a truly transparent backdrop.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setWindowTitle("SoundboardEZ")
        self.resize(920, 520)
        self.setMinimumSize(920, 520)

        self._launched_from_startup = bool(launched_from_startup)
        self._app_state: AppState = load_app_state()
        if self._app_state.startupEnabled and not is_startup_enabled():
            # Keep UI state aligned with actual registration state.
            self._app_state.startupEnabled = False
            save_app_state(self._app_state)
        self._initial_soundboard_enabled = self._determine_initial_soundboard_enabled()

        self.sounds_dir = _resolve_runtime_sounds_dir()
        self._local_all_items: list[str] = []
        self._feed_url = MYINSTANTS_INDEX_URL
        self._feed_page = 0
        self._feed_loading = False
        self._feed_end_reached = False
        self._feed_request_id = 0
        self._feed_page_cache: dict[tuple[str, int], list[tuple[str, str]]] = {}
        self._remote_seen_urls: set[str] = set()
        self._feed_play_buttons: dict[str, QPushButton] = {}
        self._pending_feed_rows: deque[RemoteSoundItem] = deque()
        self._feed_row_chunk_size = 12
        self._current_preview_url: str | None = None
        self._preview_request_id = 0
        self._threads: list[QThread] = []
        self._workers: list[QObject] = []
        self._fetch_in_progress = False
        self._route_apply_in_progress = False
        self._import_progress_dialog: QProgressDialog | None = None
        self._pending_sound_loads: list[str] = []
        self._sound_load_in_progress = False
        self._delete_mode = False
        self._trim_mode = False
        self._trim_source_path: Path | None = None
        self._trim_audio: np.ndarray | None = None
        self._trim_sr = 48000
        self._trim_playing = False
        self._trim_playhead = 0
        self._trim_decode_request_id = 0
        self._trim_stream: sd.OutputStream | None = None
        self._trim_preview_gain = 0.05
        self._trim_updating_slider = False
        self._fetch_timeout = QTimer(self)
        self._fetch_timeout.setSingleShot(True)
        self._fetch_timeout.timeout.connect(self._handle_fetch_timeout)
        self._pending_feed_scroll_value = 0
        self._feed_scroll_debounce = QTimer(self)
        self._feed_scroll_debounce.setSingleShot(True)
        self._feed_scroll_debounce.setInterval(80)
        self._feed_scroll_debounce.timeout.connect(self._load_next_feed_page_if_needed)
        self._import_search_timer = QTimer(self)
        self._import_search_timer.setSingleShot(True)
        self._import_search_timer.setInterval(280)
        self._import_search_timer.timeout.connect(self._run_debounced_import_search)
        self._feed_row_render_timer = QTimer(self)
        self._feed_row_render_timer.setSingleShot(True)
        self._feed_row_render_timer.setInterval(16)
        self._feed_row_render_timer.timeout.connect(self._drain_pending_feed_rows)
        self._local_layout_timer = QTimer(self)
        self._local_layout_timer.setSingleShot(True)
        self._local_layout_timer.setInterval(40)
        self._local_layout_timer.timeout.connect(self._apply_local_layout_refresh)
        self._last_local_tile_size = QSize()
        self._preview_monitor = QTimer(self)
        self._preview_monitor.setInterval(150)
        self._preview_monitor.timeout.connect(self._check_preview_finished)
        self._trim_play_timer = QTimer(self)
        self._trim_play_timer.setInterval(80)
        self._trim_play_timer.timeout.connect(self._sync_trim_playhead)
        self._ui_animations: list[QPropertyAnimation] = []
        self._volume_state = RuntimeVolumeState()
        self._preview_player = ManagedOutputPlayer(gain=self._volume_state.preview_gain)
        self._speaker_player = ManagedOutputPlayer(gain=self._volume_state.speaker_gain)
        self._aux_output_device_cache: int | None = None
        self._aux_output_cache_deadline = 0.0
        self._soundboard_initialized = False
        self._engine_started_event = threading.Event()
        self._engine_start_error: str | None = None
        self._tray_icon: QSystemTrayIcon | None = None
        self._tray_menu: QMenu | None = None
        self._tray_startup_action: QAction | None = None
        self._tray_start_soundboard_action: QAction | None = None
        self._tray_soundboard_power_action: QAction | None = None
        self._tray_notifications_action: QAction | None = None
        self._tray_open_action: QAction | None = None
        self._tray_quit_action: QAction | None = None
        self._quitting = False
        self._tray_hide_hint_shown = False
        self._window_corner_radius = 20
        self._window_dragging = False
        self._window_drag_offset = QPoint()
        self._window_drag_widgets: list[QObject] = []
        self._resize_margin = 8
        self._resize_edges = Qt.Edge(0)
        self._resize_origin = QPoint()
        self._resize_start_geometry = QRect()
        self._button_motion_filter: ButtonMotionFilter | None = None
        self._shell_shadow: QGraphicsDropShadowEffect | None = None

        self.engine = AudioEngine(
            samplerate=48000,
            blocksize=0,
            input_channels=1,
            output_channels=1,
            sounds_dir=str(self.sounds_dir),
        )
        self.engine.soundboard.clear_hotkeys()
        self.engine.set_mic_input_gain(self.DEFAULT_MIC_VOLUME / 100.0)
        self.engine.soundboard.set_volume(self.DEFAULT_SOUNDBOARD_VOLUME / 100.0)
        self.engine.set_noise_suppression_enabled(True)
        self.engine.set_soundboard_enabled(self._initial_soundboard_enabled)
        self._engine_thread = threading.Thread(target=self._run_engine, daemon=True)
        self._engine_thread.start()

        central = _WindowBackdrop()
        self.setCentralWidget(central)
        shell_layout = QVBoxLayout(central)
        shell_layout.setContentsMargins(12, 12, 12, 12)
        shell_layout.setSpacing(0)

        self.window_shell = RoundedContainer()
        self.window_shell.setMouseTracking(True)
        shell_layout.addWidget(self.window_shell, 1)
        self._shell_shadow = None  # shadow painted manually in _WindowBackdrop

        root_layout = QHBoxLayout(self.window_shell)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        self.main_content = QWidget()
        self.main_content.setObjectName("MainContent")
        self.main_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_column_layout = QVBoxLayout(self.main_content)
        main_column_layout.setContentsMargins(0, 0, 0, 0)
        main_column_layout.setSpacing(16)

        self.top_bar = QFrame()
        self.top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(16, 16, 16, 16)
        top_layout.setSpacing(16)
        self.app_logo = QLabel("SB")
        self.app_logo.setObjectName("LogoBadge")
        self.app_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(0)
        self.app_title = QLabel("SoundboardEZ")
        self.app_title.setObjectName("AppTitle")
        self.app_subtitle = QLabel("Easy Remote+Native Soundboard")
        self.app_subtitle.setObjectName("AppSubtitle")
        title_col.addWidget(self.app_title)
        title_col.addWidget(self.app_subtitle)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusPill")
        self.route_status_label = QLabel("Hosted On: detecting... | Mic: detecting...")
        self.route_status_label.setObjectName("RoutePill")
        self.route_status_label.setMinimumWidth(0)
        self.route_status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.window_controls = QWidget()
        self.window_controls.setObjectName("WindowControls")
        controls_row = QHBoxLayout(self.window_controls)
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(8)
        self.title_min_btn = MacTrafficButton("minimize")
        self.title_min_btn.setObjectName("TitleMinBtn")
        self.title_min_btn.setToolTip("Minimize")
        self.title_close_btn = MacTrafficButton("close")
        self.title_close_btn.setObjectName("TitleCloseBtn")
        self.title_close_btn.setToolTip("Close")
        for btn in (self.title_min_btn, self.title_close_btn):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setCheckable(False)
            controls_row.addWidget(btn)
        self.title_min_btn.clicked.connect(self.showMinimized)
        self.title_close_btn.clicked.connect(self._on_title_close_clicked)
        state_col = QVBoxLayout()
        state_col.setContentsMargins(0, 0, 0, 0)
        state_col.setSpacing(7)
        state_col.addWidget(self.window_controls, 0, Qt.AlignmentFlag.AlignRight)
        state_col.addWidget(self.status_label, 0, Qt.AlignmentFlag.AlignRight)
        state_col.addWidget(self.route_status_label, 0, Qt.AlignmentFlag.AlignRight)
        top_layout.addWidget(self.app_logo)
        top_layout.addLayout(title_col)
        top_layout.addStretch(1)
        top_layout.addLayout(state_col)
        main_column_layout.addWidget(self.top_bar)

        self.left_group = QGroupBox("")
        self.left_group.setObjectName("ImporterCard")
        self.left_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.left_group.setMinimumHeight(300)
        left_outer_layout = QVBoxLayout(self.left_group)
        left_outer_layout.setContentsMargins(0, 0, 0, 0)
        left_outer_layout.setSpacing(0)
        self.importer_scroll = QScrollArea()
        self.importer_scroll.setObjectName("ImporterScroll")
        self.importer_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.importer_scroll.setWidgetResizable(True)
        self.importer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.importer_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.importer_content = QWidget()
        self.importer_content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        left_layout = QVBoxLayout(self.importer_content)
        left_layout.setContentsMargins(24, 24, 24, 24)
        left_layout.setSpacing(24)
        self.importer_title_label = QLabel("Importer Workspace")
        self.importer_title_label.setObjectName("SectionTitle")
        self.import_search_input = QLineEdit()
        self.import_search_input.setPlaceholderText("Search sounds by name")
        self.import_search_input.setClearButtonEnabled(True)
        self.import_search_input.setMinimumHeight(40)
        self.import_search_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.import_search_input.setMinimumWidth(260)
        self.close_importer_btn = QPushButton("X")
        self.close_importer_btn.setObjectName("ImporterClose")
        self.close_importer_btn.setToolTip("Close importer")
        self.fetch_btn = QPushButton("Reload Feed")
        self.fetch_btn.setProperty("variant", "secondary")
        self.import_file_btn = QPushButton("Import File")
        self.import_file_btn.setProperty("variant", "primary")
        self.preview_volume_label = QLabel("Preview Volume: 8%")
        self.preview_volume_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.preview_volume_slider.setRange(0, 100)
        self.preview_volume_slider.setValue(8)
        self.mic_volume_label = QLabel(f"Mic Volume: {self.DEFAULT_MIC_VOLUME}%")
        self.mic_volume_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.mic_volume_slider.setRange(0, 200)
        self.mic_volume_slider.setValue(self.DEFAULT_MIC_VOLUME)
        self.remote_feed_list = SmoothListWidget(slow_factor=0.5)
        self.remote_feed_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.remote_feed_list.setSpacing(16)
        self.remote_feed_list.setMinimumHeight(120)
        self.remote_feed_list.setUniformItemSizes(True)
        self.remote_feed_list.setLayoutMode(QListView.LayoutMode.Batched)
        self.remote_feed_list.setBatchSize(48)
        self.remote_feed_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        importer_header_row = QHBoxLayout()
        importer_header_row.setContentsMargins(0, 0, 0, 0)
        importer_header_row.setSpacing(16)
        importer_header_row.addWidget(self.importer_title_label)
        importer_header_row.addStretch(1)
        importer_header_row.addWidget(self.close_importer_btn)

        importer_search_row = QHBoxLayout()
        importer_search_row.setContentsMargins(0, 0, 0, 0)
        importer_search_row.setSpacing(12)
        importer_search_row.addWidget(self.import_search_input, 1)

        importer_actions_row = QHBoxLayout()
        importer_actions_row.setContentsMargins(0, 0, 0, 0)
        importer_actions_row.setSpacing(16)
        importer_actions_row.addWidget(self.import_file_btn)
        importer_actions_row.addWidget(self.fetch_btn)
        importer_actions_row.setStretch(0, 1)
        importer_actions_row.setStretch(1, 1)

        importer_preview_row = QHBoxLayout()
        importer_preview_row.setContentsMargins(0, 0, 0, 0)
        importer_preview_row.setSpacing(16)
        importer_preview_row.addWidget(self.preview_volume_label)
        importer_preview_row.addWidget(self.preview_volume_slider, 1)

        left_layout.addLayout(importer_header_row)
        left_layout.addLayout(importer_search_row)
        left_layout.addLayout(importer_actions_row)
        left_layout.addLayout(importer_preview_row)
        importer_hint = QLabel("Click Play or Import on any sound. More loads as you scroll.")
        importer_hint.setObjectName("HintLabel")
        left_layout.addWidget(importer_hint)
        left_layout.addItem(QSpacerItem(0, 8, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        left_layout.addWidget(self.remote_feed_list, 1)
        self.importer_scroll.setWidget(self.importer_content)
        left_outer_layout.addWidget(self.importer_scroll)

        self.right_group = QGroupBox("Your Soundboard")
        self.right_group.setObjectName("MainCard")
        self.right_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout(self.right_group)
        right_layout.setContentsMargins(24, 24, 24, 24)
        right_layout.setSpacing(16)
        self.soundboard_sidebar = QWidget()
        self.soundboard_sidebar.setObjectName("SideCard")
        sidebar_w = 240
        self.soundboard_sidebar.setMinimumWidth(sidebar_w)
        self.soundboard_sidebar.setMaximumWidth(sidebar_w)
        sidebar_layout = QVBoxLayout(self.soundboard_sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(16)

        self.soundboard_main = QWidget()
        self.soundboard_main.setObjectName("SoundboardMain")
        self.soundboard_main.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_shadow = QGraphicsDropShadowEffect(self.soundboard_main)
        main_shadow.setBlurRadius(25)
        main_shadow.setOffset(0, 10)
        main_shadow.setColor(QColor(0, 0, 0, 64))
        self.soundboard_main.setGraphicsEffect(main_shadow)
        main_layout = QVBoxLayout(self.soundboard_main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(16)
        self.soundboard_volume_label = QLabel(f"Soundboard Volume: {self.DEFAULT_SOUNDBOARD_VOLUME}%")
        self.soundboard_volume_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.soundboard_volume_slider.setRange(0, 200)
        self.soundboard_volume_slider.setValue(self.DEFAULT_SOUNDBOARD_VOLUME)
        self.speaker_monitor_btn = QPushButton("Play To Speaker: Off")
        self.speaker_monitor_btn.setProperty("variant", "primary")
        self.speaker_monitor_btn.setCheckable(True)
        self.speaker_monitor_label = QLabel("Speaker Volume: 2%")
        self.speaker_monitor_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.speaker_monitor_slider.setRange(0, 100)
        self.speaker_monitor_slider.setValue(2)
        self.speaker_monitor_label.setVisible(False)
        self.speaker_monitor_slider.setVisible(False)
        self.toggle_importer_btn = QPushButton("Open Importer")
        self.toggle_importer_btn.setProperty("variant", "violet")
        self.device_label = QLabel("Audio Route")
        self.device_label.setObjectName("HintLabel")
        self.mic_device_label = QLabel("Mic Input")
        self.mic_device_combo = NoWheelComboBox()
        self.mic_device_combo.setObjectName("RouteCombo")
        self.output_device_label = QLabel("Mix Output")
        self.output_device_combo = NoWheelComboBox()
        self.output_device_combo.setObjectName("RouteCombo")
        self.refresh_devices_btn = QPushButton("Refresh Devices")
        self.refresh_devices_btn.setProperty("variant", "slate")
        self.apply_devices_btn = QPushButton("Apply Route")
        self.apply_devices_btn.setProperty("variant", "primary")
        self.mic_noise_suppression_btn = QPushButton("Mic Noise Suppression: On")
        self.mic_noise_suppression_btn.setProperty("variant", "secondary")
        self.mic_noise_suppression_btn.setCheckable(True)
        self.play_local_btn = QPushButton("Play Selected")
        self.play_local_btn.setProperty("variant", "success")
        self.trim_local_btn = QPushButton("Trim")
        self.trim_local_btn.setProperty("variant", "cyan")
        self.cancel_trim_btn = QPushButton("Cancel Trim")
        self.cancel_trim_btn.setProperty("variant", "secondary")
        self.cancel_trim_btn.setVisible(False)
        self.delete_local_btn = QPushButton("Delete Selected")
        self.delete_local_btn.setProperty("variant", "danger")
        self.cancel_delete_btn = QPushButton("Cancel Delete")
        self.cancel_delete_btn.setProperty("variant", "secondary")
        self.cancel_delete_btn.setVisible(False)
        self.refresh_local_btn = QPushButton("Refresh Local")
        self.refresh_local_btn.setProperty("variant", "slate")
        self.local_search_input = QLineEdit()
        self.local_search_input.setObjectName("LocalSearchInput")
        self.local_search_input.setPlaceholderText("Search imported sounds")
        self.local_search_input.setFixedHeight(36)
        self.local_search_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.local_list = SmoothListWidget(slow_factor=0.6)
        self.local_list.setObjectName("LocalSoundGrid")
        self.local_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.local_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.local_list.setViewMode(QListView.ViewMode.IconMode)
        self.local_list.setFlow(QListView.Flow.LeftToRight)
        self.local_list.setWrapping(True)
        self.local_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.local_list.setMovement(QListView.Movement.Static)
        self.local_list.setUniformItemSizes(True)
        self.local_list.setLayoutMode(QListView.LayoutMode.Batched)
        self.local_list.setBatchSize(120)
        self.local_list.setSpacing(12)
        self.local_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.local_list.setWordWrap(True)
        self.local_list.setSelectionRectVisible(False)
        self.local_list.setMouseTracking(True)
        self.local_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.local_list.customContextMenuRequested.connect(self._show_local_context_menu)
        self.local_list.setItemDelegate(LocalTileDelegate(self.local_list))
        self.local_title = QLabel("Imported Sounds--ami_nope")
        self.local_title.setObjectName("SectionTitle")
        title_font = QFont(self.local_title.font())
        title_font.setPointSize(22)
        title_font.setWeight(QFont.Weight.DemiBold)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -0.3)
        self.local_title.setFont(title_font)
        self._local_tile_colors: dict[str, tuple[str, str]] = {}
        self.delete_mode_hint = QLabel("Delete mode: tick sounds, then click Delete Checked")
        self.delete_mode_hint.setObjectName("HintLabel")
        self.delete_mode_hint.setVisible(False)
        self.trim_mode_hint = QLabel("Trim mode: click a sound tile to open the trim window.")
        self.trim_mode_hint.setObjectName("HintLabel")
        self.trim_mode_hint.setVisible(False)
        self.trim_dialog = TrimEditorDialog(self)
        self.trim_dialog.hide()
        self.trim_dialog.closed.connect(self.close_trim_editor)
        self.trim_play_pause_btn = self.trim_dialog.trim_play_pause_btn
        self.trim_stop_btn = self.trim_dialog.trim_stop_btn
        self.apply_trim_btn = self.trim_dialog.apply_trim_btn
        self.close_trim_editor_btn = self.trim_dialog.close_trim_editor_btn
        self._delete_hint_fx = QGraphicsOpacityEffect(self.delete_mode_hint)
        self.delete_mode_hint.setGraphicsEffect(self._delete_hint_fx)
        self._delete_hint_anim = QPropertyAnimation(self._delete_hint_fx, b"opacity", self)
        self._delete_hint_anim.setDuration(250)
        self._delete_hint_anim.setStartValue(0.0)
        self._delete_hint_anim.setEndValue(1.0)

        route_section_layout = QVBoxLayout()
        route_section_layout.setContentsMargins(0, 0, 0, 0)
        route_section_layout.setSpacing(16)
        route_section_layout.addWidget(self.device_label)

        mic_route_layout = QVBoxLayout()
        mic_route_layout.setContentsMargins(0, 0, 0, 0)
        mic_route_layout.setSpacing(8)
        mic_route_layout.addWidget(self.mic_device_label)
        mic_route_layout.addWidget(self.mic_device_combo)
        route_section_layout.addLayout(mic_route_layout)

        output_route_layout = QVBoxLayout()
        output_route_layout.setContentsMargins(0, 0, 0, 0)
        output_route_layout.setSpacing(8)
        output_route_layout.addWidget(self.output_device_label)
        output_route_layout.addWidget(self.output_device_combo)
        route_section_layout.addLayout(output_route_layout)

        route_actions_layout = QVBoxLayout()
        route_actions_layout.setContentsMargins(0, 0, 0, 0)
        route_actions_layout.setSpacing(16)
        route_actions_layout.addWidget(self.refresh_devices_btn)
        route_actions_layout.addWidget(self.apply_devices_btn)
        route_actions_layout.addWidget(self.mic_noise_suppression_btn)
        route_section_layout.addLayout(route_actions_layout)

        volume_section_layout = QVBoxLayout()
        volume_section_layout.setContentsMargins(0, 0, 0, 0)
        volume_section_layout.setSpacing(8)
        volume_section_layout.addWidget(self.soundboard_volume_label)
        volume_section_layout.addWidget(self.soundboard_volume_slider)
        volume_section_layout.addWidget(self.mic_volume_label)
        volume_section_layout.addWidget(self.mic_volume_slider)

        speaker_section_layout = QVBoxLayout()
        speaker_section_layout.setContentsMargins(0, 0, 0, 0)
        speaker_section_layout.setSpacing(16)
        speaker_section_layout.addWidget(self.speaker_monitor_btn)

        speaker_volume_layout = QVBoxLayout()
        speaker_volume_layout.setContentsMargins(0, 0, 0, 0)
        speaker_volume_layout.setSpacing(8)
        speaker_volume_layout.addWidget(self.speaker_monitor_label)
        speaker_volume_layout.addWidget(self.speaker_monitor_slider)
        speaker_section_layout.addLayout(speaker_volume_layout)

        actions_section_layout = QVBoxLayout()
        actions_section_layout.setContentsMargins(0, 0, 0, 0)
        actions_section_layout.setSpacing(16)
        actions_section_layout.addWidget(self.play_local_btn)
        actions_section_layout.addWidget(self.trim_local_btn)
        actions_section_layout.addWidget(self.cancel_trim_btn)
        actions_section_layout.addWidget(self.delete_local_btn)
        actions_section_layout.addWidget(self.cancel_delete_btn)
        actions_section_layout.addWidget(self.refresh_local_btn)

        hint_section_layout = QVBoxLayout()
        hint_section_layout.setContentsMargins(0, 0, 0, 0)
        hint_section_layout.setSpacing(8)
        hint_section_layout.addWidget(self.delete_mode_hint)
        hint_section_layout.addWidget(self.trim_mode_hint)

        sidebar_layout.addWidget(self.toggle_importer_btn)
        sidebar_layout.addItem(QSpacerItem(0, 24, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        sidebar_layout.addLayout(route_section_layout)
        sidebar_layout.addItem(QSpacerItem(0, 24, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        sidebar_layout.addLayout(volume_section_layout)
        sidebar_layout.addItem(QSpacerItem(0, 24, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        sidebar_layout.addLayout(speaker_section_layout)
        sidebar_layout.addItem(QSpacerItem(0, 24, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        sidebar_layout.addLayout(actions_section_layout)
        sidebar_layout.addItem(QSpacerItem(0, 24, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        sidebar_layout.addLayout(hint_section_layout)
        sidebar_layout.addStretch(1)

        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setObjectName("SideScroll")
        self.sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.sidebar_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.sidebar_scroll.setWidget(self.soundboard_sidebar)
        self.sidebar_scroll.verticalScrollBar().setSingleStep(14)

        self.sidebar_shell = QFrame()
        self.sidebar_shell.setObjectName("SideCardShell")
        self.sidebar_shell.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.sidebar_shell.setMinimumWidth(sidebar_w + 10)
        self.sidebar_shell.setMaximumWidth(sidebar_w + 10)
        sidebar_shell_layout = QVBoxLayout(self.sidebar_shell)
        sidebar_shell_layout.setContentsMargins(1, 1, 1, 1)
        sidebar_shell_layout.setSpacing(0)
        sidebar_shell_layout.addWidget(self.sidebar_scroll)

        local_header_shell = QFrame()
        local_header_shell.setObjectName("LocalHeaderShell")
        header_shell_layout = QHBoxLayout(local_header_shell)
        header_shell_layout.setContentsMargins(14, 10, 14, 10)
        header_shell_layout.setSpacing(14)
        header_shell_layout.addWidget(self.local_title)
        header_shell_layout.addStretch(1)
        header_shell_layout.addWidget(self.local_search_input, 0)
        main_layout.addWidget(local_header_shell)

        self.local_list_shell = QFrame()
        self.local_list_shell.setObjectName("LocalListShell")
        shell_layout = QVBoxLayout(self.local_list_shell)
        shell_layout.setContentsMargins(20, 20, 20, 20)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(self.local_list)
        main_layout.addWidget(self.local_list_shell, 1)

        right_layout.addWidget(self.soundboard_main, 1)
        main_column_layout.addWidget(self.right_group, 1)
        root_layout.addWidget(self.sidebar_shell)
        root_layout.addWidget(self.main_content, 1)

        self.importer_window = FramelessImporterDialog(self)
        self.importer_window.setObjectName("ImporterWindow")
        self.importer_window.setWindowTitle("Importer - SoundboardEZ")
        self.importer_window.setModal(False)
        self.importer_window.setMinimumSize(680, 520)
        self.importer_window.resize(760, 620)
        self.importer_shell = QWidget()
        self.importer_shell.setObjectName("ImporterShell")
        self.importer_shell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        importer_shadow = QGraphicsDropShadowEffect(self.importer_shell)
        importer_shadow.setBlurRadius(28)
        importer_shadow.setOffset(0, 8)
        importer_shadow.setColor(QColor(4, 11, 24, 160))
        importer_shadow.setEnabled(True)
        self.importer_shell.setGraphicsEffect(importer_shadow)
        importer_shell_layout = QVBoxLayout(self.importer_shell)
        importer_shell_layout.setContentsMargins(20, 20, 20, 20)
        importer_shell_layout.setSpacing(0)
        importer_shell_layout.addWidget(self.left_group)
        importer_window_layout = QVBoxLayout(self.importer_window)
        # Margins give the rounded corners + drop shadow room to breathe.
        importer_window_layout.setContentsMargins(16, 12, 16, 20)
        importer_window_layout.setSpacing(0)
        importer_window_layout.addWidget(self.importer_shell)
        self.importer_window.finished.connect(lambda _=0: self._set_importer_visible(False))

        self.close_importer_btn.clicked.connect(self.close_importer_panel)
        self.import_search_input.returnPressed.connect(self.apply_import_search)
        self.import_search_input.textChanged.connect(self._schedule_import_search)
        self.fetch_btn.clicked.connect(self.fetch_sounds)
        self.preview_volume_slider.valueChanged.connect(self.update_preview_volume_label)
        self.mic_volume_slider.valueChanged.connect(self.update_mic_volume_label)
        self.soundboard_volume_slider.valueChanged.connect(self.update_soundboard_volume)
        self.speaker_monitor_btn.toggled.connect(self.toggle_speaker_monitor)
        self.mic_noise_suppression_btn.toggled.connect(self.toggle_mic_noise_suppression)
        self.speaker_monitor_slider.valueChanged.connect(self.update_speaker_monitor_volume_label)
        self.refresh_devices_btn.clicked.connect(self.refresh_audio_devices)
        self.apply_devices_btn.clicked.connect(self.apply_audio_route)
        self.toggle_importer_btn.clicked.connect(self.toggle_importer_panel)
        self.play_local_btn.clicked.connect(self.play_selected_imported)
        self.trim_local_btn.clicked.connect(self.trim_selected_imported)
        self.cancel_trim_btn.clicked.connect(self.cancel_trim_mode)
        self.delete_local_btn.clicked.connect(self.delete_selected_imported)
        self.cancel_delete_btn.clicked.connect(self.cancel_delete_mode)
        self.apply_trim_btn.clicked.connect(self.apply_trim_from_editor)
        self.close_trim_editor_btn.clicked.connect(self.close_trim_editor)
        self.trim_play_pause_btn.clicked.connect(self.toggle_trim_preview_play)
        self.trim_stop_btn.clicked.connect(lambda _=False: self.stop_trim_preview(reset_to_start=True))
        self.trim_dialog.timeline.playheadChanged.connect(self._on_trim_timeline_changed)
        self.trim_dialog.timeline.startChanged.connect(self._on_trim_range_changed)
        self.trim_dialog.timeline.endChanged.connect(self._on_trim_range_changed)
        self.import_file_btn.clicked.connect(self.import_from_file_dialog)
        self.refresh_local_btn.clicked.connect(self.refresh_local)
        self.local_search_input.textChanged.connect(self.apply_local_filter)
        self.local_list.itemDoubleClicked.connect(self.play_imported_item)
        self.local_list.itemClicked.connect(self._on_local_item_clicked)
        self.remote_feed_list.verticalScrollBar().valueChanged.connect(self._on_feed_scroll)

        self._apply_modern_theme()
        self._apply_button_ratios()
        self._apply_button_motion()
        self._install_window_interaction_regions()
        self._setup_desktop_notifications()
        self._importer_loaded_once = False
        self._set_importer_visible(False)
        self.refresh_audio_devices()
        self.refresh_local()
        self.update_preview_volume_label(self.preview_volume_slider.value())
        self.update_mic_volume_label(self.mic_volume_slider.value())
        self.update_soundboard_volume(self.soundboard_volume_slider.value())
        self.update_speaker_monitor_volume_label(self.speaker_monitor_slider.value())
        self.engine.set_noise_suppression_enabled(True)
        self._set_soundboard_enabled(self._initial_soundboard_enabled, persist=False, silent=True)
        self._sync_mic_noise_suppression_button()
        self._sync_startup_tray_state()
        self._update_route_status()
        self._apply_window_mask()
        _apply_dwm_rounded_corners(self)
        self._run_entrance_animation()

    def _run_engine(self) -> None:
        self._engine_start_error = None
        self._engine_started_event.clear()

        def _on_engine_started() -> None:
            self._engine_started_event.set()

        try:
            if not self._soundboard_initialized:
                self.engine.setup_soundboard(auto_hotkeys=False)
                self.engine.soundboard.clear_hotkeys()
                self._soundboard_initialized = True
            self.engine.start(on_started=_on_engine_started)
        except Exception as exc:
            self._engine_start_error = str(exc)
            self._engine_started_event.set()
            print(f"Audio engine error: {exc}")

    @staticmethod
    def _coerce_device_data(value) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if not text:
            return None
        if text == "AUTO":
            return None
        try:
            return int(text)
        except Exception:
            return None

    def _update_route_status(self) -> None:
        in_name, out_name = self.engine.get_route_summary()
        self.route_status_label.setText(f"Hosted On: {out_name} | Mic: {in_name}")

    def refresh_audio_devices(self) -> None:
        self._aux_output_cache_deadline = 0.0
        current_in = self._coerce_device_data(self.mic_device_combo.currentData())
        current_out = self._coerce_device_data(self.output_device_combo.currentData())

        self.mic_device_combo.clear()
        self.output_device_combo.clear()
        self.mic_device_combo.addItem("System Default", "AUTO")
        self.output_device_combo.addItem("Auto (VB-Cable)", "AUTO")

        try:
            for idx, label in self.engine.list_input_devices():
                self.mic_device_combo.addItem(label, idx)
            for idx, label in self.engine.list_output_devices():
                self.output_device_combo.addItem(label, idx)
        except Exception as exc:
            self.status_label.setText(f"Device query failed: {exc}")
            return

        target_in = self.engine.input_device if self.engine.input_device is not None else current_in
        target_out = self.engine.output_device if self.engine.output_device is not None else current_out
        if target_in is not None:
            idx = self.mic_device_combo.findData(target_in)
            if idx >= 0:
                self.mic_device_combo.setCurrentIndex(idx)
        if target_out is not None:
            idx = self.output_device_combo.findData(target_out)
            if idx >= 0:
                self.output_device_combo.setCurrentIndex(idx)

        self.status_label.setText("Audio devices refreshed.")
        self._update_route_status()

    def _restart_audio_engine(self, start_timeout: float = 4.0) -> tuple[bool, str | None]:
        self.engine.stop()
        if self._engine_thread.is_alive():
            self._engine_thread.join(timeout=5.0)
        if self._engine_thread.is_alive():
            return False, "Previous audio engine thread did not stop."

        self._engine_start_error = None
        self._engine_started_event.clear()
        self._engine_thread = threading.Thread(target=self._run_engine, daemon=True)
        self._engine_thread.start()
        deadline = time.monotonic() + max(0.2, float(start_timeout))
        while time.monotonic() < deadline:
            if self._engine_started_event.wait(timeout=0.05):
                break
            if not self._engine_thread.is_alive():
                break

        if self._engine_start_error:
            return False, self._engine_start_error
        if self._engine_started_event.is_set():
            return True, None
        if not self._engine_thread.is_alive():
            return False, "Audio engine thread exited before startup."
        return False, "Audio engine start timed out."

    def apply_audio_route(self) -> None:
        if self._route_apply_in_progress:
            self.status_label.setText("Route change already in progress...")
            return

        in_dev = self._coerce_device_data(self.mic_device_combo.currentData())
        out_dev = self._coerce_device_data(self.output_device_combo.currentData())
        old_in = self.engine.input_device
        old_out = self.engine.output_device

        if in_dev == old_in and out_dev == old_out:
            self.status_label.setText("Audio route unchanged.")
            self._update_route_status()
            return

        self.stop_remote_preview(silent=True)
        self._speaker_player.stop()
        self._route_apply_in_progress = True
        self.apply_devices_btn.setEnabled(False)
        self.status_label.setText("Applying audio route...")

        worker = RouteApplyWorker(
            engine=self.engine,
            restart_fn=self._restart_audio_engine,
            input_device=in_dev,
            output_device=out_dev,
            old_input_device=old_in,
            old_output_device=old_out,
        )

        def done(result: dict) -> None:
            self._route_apply_in_progress = False
            self.apply_devices_btn.setEnabled(True)
            if bool(result.get("ok")):
                self._aux_output_cache_deadline = 0.0
                msg = "Audio route applied."
                self.status_label.setText(msg)
                self._notify_desktop(msg, title="Audio Route")
                self._update_route_status()
                return

            err = result.get("err")
            rollback_ok = bool(result.get("rollback_ok"))
            rollback_err = result.get("rollback_err")
            if rollback_ok:
                self._aux_output_cache_deadline = 0.0
                msg = f"Route failed ({err}). Reverted to previous route."
                self.status_label.setText(msg)
                self._notify_desktop(msg, title="Audio Route", error=True)
                self._update_route_status()
                return

            msg = f"Route failed ({err}). Restore failed ({rollback_err})."
            self.status_label.setText(msg)
            self._notify_desktop(msg, title="Audio Route", error=True)
            self._update_route_status()

        def err(message: str) -> None:
            self._route_apply_in_progress = False
            self.apply_devices_btn.setEnabled(True)
            self.status_label.setText(f"Route failed: {message}")
            self._notify_desktop(self.status_label.text(), title="Audio Route", error=True)
            self._update_route_status()

        self._run_worker(worker, done, err)

    def _apply_button_ratios(self) -> None:
        sidebar_buttons = [
            self.toggle_importer_btn,
            self.speaker_monitor_btn,
            self.mic_noise_suppression_btn,
            self.refresh_devices_btn,
            self.apply_devices_btn,
            self.play_local_btn,
            self.trim_local_btn,
            self.cancel_trim_btn,
            self.delete_local_btn,
            self.cancel_delete_btn,
            self.refresh_local_btn,
        ]
        for btn in sidebar_buttons:
            btn.setMinimumHeight(34)
            btn.setMaximumHeight(36)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        importer_buttons = [
            self.fetch_btn,
            self.import_file_btn,
        ]
        for btn in importer_buttons:
            btn.setMinimumHeight(40)
            btn.setMaximumHeight(44)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        self.close_importer_btn.setMinimumSize(32, 32)
        self.close_importer_btn.setMaximumSize(32, 32)
        self.local_search_input.setMinimumWidth(220)
        self.import_search_input.setMinimumHeight(46)
        self.preview_volume_slider.setMinimumWidth(220)
        self.mic_volume_slider.setMinimumWidth(0)
        self.mic_volume_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.mic_device_combo.setMinimumHeight(32)
        self.mic_device_combo.setMinimumWidth(0)
        self.mic_device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.output_device_combo.setMinimumHeight(32)
        self.output_device_combo.setMinimumWidth(0)
        self.output_device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _apply_modern_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget#WindowBackdrop {
                background: transparent;
            }
            QFrame#RoundedContainer {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #111f33,
                    stop:1 #152a44);
                border: none;
                border-radius: 22px;
                color: #e7edf7;
                font-family: "Inter", "Segoe UI Variable Text", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QWidget#MainContent {
                background: transparent;
            }
            QDialog#ImporterWindow {
                background: transparent;
                border: none;
            }
            QWidget#ImporterShell {
                background: rgba(19, 33, 52, 236);
                border: none;
                border-radius: 28px;
            }
            QFrame#TopBar {
                background: rgba(49, 70, 97, 210);
                border: 1px solid rgba(162, 183, 209, 40);
                border-radius: 20px;
            }
            QWidget#WindowControls {
                background: transparent;
            }
            QPushButton#TitleMinBtn, QPushButton#TitleCloseBtn {
                min-width: 16px;
                max-width: 16px;
                min-height: 16px;
                max-height: 16px;
                padding: 0px;
                border: none;
                background: transparent;
            }
            QLabel#LogoBadge {
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
                border-radius: 16px;
                background: rgba(16, 27, 45, 196);
                border: 1px solid #8ad8ff;
                color: #e7edf7;
                font-size: 12px;
                font-weight: 650;
            }
            QLabel#AppTitle {
                color: #f1f5fb;
                font-size: 24px;
                font-weight: 600;
                letter-spacing: 0.2px;
            }
            QLabel#AppSubtitle {
                color: #aac2df;
                font-size: 15px;
                font-weight: 500;
            }
            QGroupBox#MainCard {
                background: rgba(38, 55, 78, 214);
                border: 1px solid rgba(170, 191, 217, 20);
                border-radius: 22px;
                margin-top: 18px;
                padding-top: 14px;
            }
            QGroupBox#MainCard::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 2px 12px;
                color: #d5e1f1;
                font-size: 14px;
                font-weight: 600;
                background: rgba(19, 30, 47, 210);
                border: none;
                border-radius: 11px;
            }
            QGroupBox#ImporterCard {
                background: transparent;
                border: none;
                margin-top: 0px;
                padding-top: 0px;
            }
            QGroupBox#ImporterCard::title {
                width: 0px;
                height: 0px;
                padding: 0px;
            }
            QFrame#SideCardShell {
                background: rgba(47, 66, 92, 182);
                border: 1px solid rgba(164, 185, 212, 32);
                border-radius: 20px;
            }
            QWidget#SideCard {
                background: transparent;
                border: none;
                border-radius: 0px;
            }
            QScrollArea#SideScroll {
                background: transparent;
                border: none;
            }
            QScrollArea#ImporterScroll {
                background: transparent;
                border: none;
            }
            QScrollArea#SideScroll QWidget#qt_scrollarea_viewport {
                background: transparent;
                border: none;
            }
            QScrollArea#ImporterScroll QWidget#qt_scrollarea_viewport {
                background: transparent;
                border: none;
            }
            QWidget#SoundboardMain {
                background: rgba(51, 72, 99, 226);
                border: 1px solid rgba(166, 186, 211, 16);
                border-radius: 20px;
                padding: 24px;
            }
            QLabel#SectionTitle {
                font-size: 22px;
                font-weight: 600;
                color: #edf3fb;
                letter-spacing: -0.3px;
            }
            QLabel {
                color: #e7edf7;
            }
            QLabel#HintLabel {
                color: #8fa5c1;
                font-size: 12px;
            }
            QLabel#FeedNameLabel {
                color: #e6eef8;
                font-size: 14px;
                font-weight: 500;
            }
            QWidget#FeedRow {
                background: rgba(17, 28, 46, 182);
                border: 1px solid rgba(148, 163, 184, 26);
                border-radius: 16px;
            }
            QLabel#StatusPill {
                background: rgba(21, 33, 52, 210);
                color: #d8e4f2;
                border: 1px solid rgba(148, 163, 184, 34);
                border-radius: 13px;
                padding: 7px 14px;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel#RoutePill {
                background: transparent;
                color: #9fb3cb;
                border: none;
                padding: 0px 2px;
                font-size: 12px;
                font-weight: 500;
            }
            QLineEdit {
                background: rgba(8, 18, 33, 235);
                border: 1px solid rgba(148, 163, 184, 44);
                border-radius: 15px;
                padding: 10px 13px;
                selection-background-color: #59d7ff;
                color: #e7edf7;
                font-size: 14px;
            }
            QLineEdit#LocalSearchInput {
                color: rgba(231, 237, 247, 204);
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #6be0ff;
                background: rgba(10, 24, 43, 245);
            }
            QComboBox#RouteCombo {
                background: rgba(9, 19, 34, 235);
                border: 1px solid rgba(148, 163, 184, 44);
                border-radius: 15px;
                padding: 8px 10px;
                color: #e7edf7;
                font-size: 14px;
            }
            QComboBox#RouteCombo::drop-down {
                border: none;
                width: 22px;
            }
            QComboBox#RouteCombo QAbstractItemView {
                background: #0f1b2d;
                border: 1px solid #3b4d68;
                border-radius: 12px;
                color: #e7edf7;
                selection-background-color: #294768;
            }
            QPushButton {
                border-radius: 15px;
                border: 1px solid rgba(148, 163, 184, 52);
                padding: 6px 14px;
                min-height: 30px;
                background: rgba(70, 90, 118, 224);
                color: #eaf2fb;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background: rgba(84, 108, 138, 232);
                border-color: rgba(107, 224, 255, 196);
            }
            QPushButton:pressed {
                background: rgba(60, 80, 104, 238);
            }
            QPushButton:checked,
            QPushButton[active="true"] {
                background: #1084b7;
                border-color: #74e1ff;
                color: #f7fbff;
            }
            QPushButton[variant="danger"] {
                background: rgba(111, 43, 57, 222);
                border-color: rgba(248, 113, 113, 128);
            }
            QPushButton[variant="danger"]:hover {
                background: rgba(132, 52, 69, 232);
                border-color: rgba(252, 165, 165, 188);
            }
            QPushButton#ImporterClose {
                min-width: 28px;
                max-width: 28px;
                min-height: 28px;
                max-height: 28px;
                border-radius: 14px;
                padding: 0;
                background: rgba(69, 85, 110, 218);
                border: 1px solid rgba(148, 163, 184, 52);
                color: #f0f6ff;
                font-weight: 700;
            }
            QPushButton#ImporterClose:hover {
                background: rgba(86, 104, 134, 230);
                border-color: rgba(96, 214, 255, 196);
            }
            QListWidget {
                background: rgba(10, 21, 38, 228);
                border: 1px solid rgba(148, 163, 184, 36);
                border-radius: 18px;
                padding: 10px;
                outline: none;
            }
            QFrame#LocalListShell {
                background: rgba(16, 24, 36, 210);
                border: 1px solid rgba(255, 255, 255, 16);
                border-radius: 20px;
            }
            QFrame#LocalHeaderShell {
                background: rgba(240, 244, 248, 32);
                border: 1px solid rgba(255, 255, 255, 26);
                border-radius: 18px;
            }
            QListWidget#LocalSoundGrid {
                background: rgba(18, 26, 38, 196);
                border: 1px solid rgba(255, 255, 255, 18);
                border-radius: 16px;
                padding: 12px;
            }
            QListWidget#LocalSoundGrid:hover {
                border-color: rgba(255, 255, 255, 30);
            }
            QListWidget#LocalSoundGrid::item {
                border: none;
                color: #f2f7ff;
                border-radius: 14px;
            }
            QListWidget#LocalSoundGrid::item:selected {
                background: transparent;
                color: #ffffff;
            }
            QMenu#LocalTileMenu {
                background: rgba(18, 26, 38, 230);
                border: 1px solid rgba(148, 163, 184, 40);
                border-radius: 10px;
                padding: 6px 8px;
            }
            QMenu#LocalTileMenu::item {
                color: #eaf2fb;
                padding: 6px 10px;
                border-radius: 8px;
                min-width: 140px;
            }
            QMenu#LocalTileMenu::item:selected {
                background: rgba(80, 130, 200, 170);
                color: #ffffff;
            }
            QListWidget::item {
                border-radius: 12px;
                padding: 6px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background: rgba(33, 52, 75, 210);
                color: #edf4fc;
            }
            QSlider::groove:horizontal {
                border-radius: 6px;
                height: 8px;
                background: rgba(26, 42, 62, 220);
            }
            QSlider::sub-page:horizontal {
                border-radius: 6px;
                background: #59d7ff;
            }
            QSlider::handle:horizontal {
                background: #dbe7f6;
                border: 1px solid #6b7f9b;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 11px;
                margin: 3px;
            }
            QScrollBar::handle:vertical {
                background: rgba(85, 104, 132, 220);
                border: 1px solid rgba(148, 163, 184, 76);
                border-radius: 5px;
                min-height: 34px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QPushButton#FeedButton {
                border-radius: 12px;
                border: 1px solid rgba(148, 163, 184, 52);
                font-size: 13px;
                font-weight: 600;
                min-height: 0px;
                max-height: 32px;
                padding: 0px 10px;
                color: #eaf2fb;
                background: rgba(64, 82, 108, 220);
            }
            QPushButton#FeedButton:hover {
                background: rgba(81, 101, 130, 230);
                border-color: rgba(96, 214, 255, 188);
            }
            QPushButton#FeedButton:pressed {
                background: rgba(58, 77, 102, 235);
            }
            QPushButton#FeedButton[active="true"] {
                background: #0f6f98;
                border-color: #62d6ff;
                color: #f7fbff;
            }
            QPushButton#FeedButton:disabled {
                background: rgba(27, 39, 58, 220);
                border-color: rgba(71, 85, 105, 120);
                color: #8fa5c1;
            }
            """
        )

    def _apply_button_motion(self) -> None:
        # Title traffic-light buttons render/animate themselves.
        return

    def _install_window_interaction_regions(self) -> None:
        drag_widgets = [
            self.top_bar,
            self.app_logo,
            self.app_title,
            self.app_subtitle,
            self.status_label,
            self.route_status_label,
        ]
        for widget in drag_widgets:
            if widget in self._window_drag_widgets:
                continue
            self._window_drag_widgets.append(widget)
            widget.installEventFilter(self)
        self.window_shell.installEventFilter(self)

    @staticmethod
    def _is_drag_exempt_widget(widget: QWidget | None) -> bool:
        current = widget
        while current is not None:
            if isinstance(
                current,
                (QPushButton, QLineEdit, QComboBox, QSlider, QListWidget, QAbstractItemView, QScrollArea),
            ):
                return True
            current = current.parentWidget()
        return False

    def _hit_test_resize_edges(self, pos: QPoint) -> Qt.Edge:
        if self.isMaximized():
            return Qt.Edge(0)
        rect = self.window_shell.rect()
        margin = max(4, int(self._resize_margin))
        if rect.width() < (margin * 2) or rect.height() < (margin * 2):
            return Qt.Edge(0)

        edges = Qt.Edge(0)
        if pos.x() <= margin:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() >= rect.width() - margin:
            edges |= Qt.Edge.RightEdge

        if pos.y() <= margin:
            edges |= Qt.Edge.TopEdge
        elif pos.y() >= rect.height() - margin:
            edges |= Qt.Edge.BottomEdge
        return edges

    @staticmethod
    def _cursor_for_edges(edges: Qt.Edge) -> Qt.CursorShape:
        if edges == Qt.Edge(0):
            return Qt.CursorShape.ArrowCursor
        has_left = bool(edges & Qt.Edge.LeftEdge)
        has_right = bool(edges & Qt.Edge.RightEdge)
        has_top = bool(edges & Qt.Edge.TopEdge)
        has_bottom = bool(edges & Qt.Edge.BottomEdge)
        if (has_left and has_top) or (has_right and has_bottom):
            return Qt.CursorShape.SizeFDiagCursor
        if (has_right and has_top) or (has_left and has_bottom):
            return Qt.CursorShape.SizeBDiagCursor
        if has_left or has_right:
            return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.SizeVerCursor

    def _resize_from_edges(self, global_pos: QPoint) -> None:
        if self._resize_edges == Qt.Edge(0):
            return
        delta = global_pos - self._resize_origin
        geo = QRect(self._resize_start_geometry)

        if self._resize_edges & Qt.Edge.LeftEdge:
            geo.setLeft(geo.left() + delta.x())
        if self._resize_edges & Qt.Edge.RightEdge:
            geo.setRight(geo.right() + delta.x())
        if self._resize_edges & Qt.Edge.TopEdge:
            geo.setTop(geo.top() + delta.y())
        if self._resize_edges & Qt.Edge.BottomEdge:
            geo.setBottom(geo.bottom() + delta.y())

        min_w = max(1, self.minimumWidth())
        min_h = max(1, self.minimumHeight())

        if geo.width() < min_w:
            if self._resize_edges & Qt.Edge.LeftEdge:
                geo.setLeft(geo.right() - min_w + 1)
            else:
                geo.setRight(geo.left() + min_w - 1)
        if geo.height() < min_h:
            if self._resize_edges & Qt.Edge.TopEdge:
                geo.setTop(geo.bottom() - min_h + 1)
            else:
                geo.setBottom(geo.top() + min_h - 1)
        self.setGeometry(geo)

    def _build_squircle_path(self, rect: QRectF, radius: float) -> QPainterPath:
        r = max(0.0, min(float(radius), rect.width() * 0.5, rect.height() * 0.5))
        c = r * 0.42
        path = QPainterPath()
        path.moveTo(rect.left() + r, rect.top())
        path.lineTo(rect.right() - r, rect.top())
        path.cubicTo(
            rect.right() - c,
            rect.top(),
            rect.right(),
            rect.top() + c,
            rect.right(),
            rect.top() + r,
        )
        path.lineTo(rect.right(), rect.bottom() - r)
        path.cubicTo(
            rect.right(),
            rect.bottom() - c,
            rect.right() - c,
            rect.bottom(),
            rect.right() - r,
            rect.bottom(),
        )
        path.lineTo(rect.left() + r, rect.bottom())
        path.cubicTo(
            rect.left() + c,
            rect.bottom(),
            rect.left(),
            rect.bottom() - c,
            rect.left(),
            rect.bottom() - r,
        )
        path.lineTo(rect.left(), rect.top() + r)
        path.cubicTo(
            rect.left(),
            rect.top() + c,
            rect.left() + c,
            rect.top(),
            rect.left() + r,
            rect.top(),
        )
        path.closeSubpath()
        return path

    def _apply_window_mask(self) -> None:
        if not hasattr(self, "window_shell"):
            return
        # Keep native rectangular window for stable GPU composition and hit-testing.
        # Rounded visuals are handled by the styled RoundedContainer.
        self.clearMask()
        self.window_shell.clearMask()

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if obj is self.window_shell:
            typ = event.type()
            if typ == QEvent.Type.MouseMove:
                if self._resize_edges != Qt.Edge(0) and (event.buttons() & Qt.MouseButton.LeftButton):
                    self._resize_from_edges(event.globalPosition().toPoint())
                    return True
                if self._window_dragging and (event.buttons() & Qt.MouseButton.LeftButton):
                    self.move(event.globalPosition().toPoint() - self._window_drag_offset)
                    return True
                edges = self._hit_test_resize_edges(event.position().toPoint())
                self.window_shell.setCursor(self._cursor_for_edges(edges))
            elif typ == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    edges = self._hit_test_resize_edges(event.position().toPoint())
                    if edges != Qt.Edge(0):
                        self._resize_edges = edges
                        self._resize_origin = event.globalPosition().toPoint()
                        self._resize_start_geometry = self.geometry()
                        return True
                    if not self.isMaximized():
                        global_pos = event.globalPosition().toPoint()
                        local_pos = self.mapFromGlobal(global_pos)
                        child = self.childAt(local_pos)
                        if not self._is_drag_exempt_widget(child):
                            self._window_dragging = True
                            self._window_drag_offset = global_pos - self.frameGeometry().topLeft()
                            return True
            elif typ == QEvent.Type.MouseButtonRelease:
                if self._resize_edges != Qt.Edge(0):
                    self._resize_edges = Qt.Edge(0)
                    self.window_shell.setCursor(Qt.CursorShape.ArrowCursor)
                    return True
                if self._window_dragging:
                    self._window_dragging = False
                    self.window_shell.setCursor(Qt.CursorShape.ArrowCursor)
                    return True
            elif typ == QEvent.Type.Leave and self._resize_edges == Qt.Edge(0):
                if not self._window_dragging:
                    self.window_shell.setCursor(Qt.CursorShape.ArrowCursor)

        if obj in self._window_drag_widgets:
            typ = event.type()
            if typ == QEvent.Type.MouseButtonPress:
                if event.button() != Qt.MouseButton.LeftButton or self.isMaximized():
                    return super().eventFilter(obj, event)
                global_pos = event.globalPosition().toPoint()
                local_pos = self.mapFromGlobal(global_pos)
                child = self.childAt(local_pos)
                if self._is_drag_exempt_widget(child):
                    return super().eventFilter(obj, event)
                self._window_dragging = True
                self._window_drag_offset = global_pos - self.frameGeometry().topLeft()
                return True
            if typ == QEvent.Type.MouseMove and self._window_dragging and (event.buttons() & Qt.MouseButton.LeftButton):
                self.move(event.globalPosition().toPoint() - self._window_drag_offset)
                return True
            if typ == QEvent.Type.MouseButtonRelease and self._window_dragging:
                self._window_dragging = False
                return True

        return super().eventFilter(obj, event)

    def _run_entrance_animation(self) -> None:
        sequence = [
            (self.top_bar, 0),
            (self.sidebar_shell, 70),
            (self.right_group, 120),
        ]
        for widget, start_ms in sequence:
            if not widget.isVisible():
                continue
            fx = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(fx)
            fx.setOpacity(0.0)
            anim = QPropertyAnimation(fx, b"opacity", self)
            anim.setDuration(460)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            # Remove the QGraphicsOpacityEffect once the animation finishes
            # so the widget no longer needs off-screen compositing.
            w_ref = widget
            anim.finished.connect(lambda w=w_ref: w.setGraphicsEffect(None))
            self._ui_animations.append(anim)
            QTimer.singleShot(start_ms, anim.start)

    def _determine_initial_soundboard_enabled(self) -> bool:
        if self._launched_from_startup and self._app_state.startupEnabled:
            return bool(self._app_state.startSoundboardOnLaunch)
        return bool(self._app_state.soundboardEnabled)

    def _persist_app_state(self) -> None:
        self._app_state.soundboardEnabled = bool(self.engine.is_soundboard_enabled())
        save_app_state(self._app_state)

    @staticmethod
    def _set_action_checked(action: QAction | None, checked: bool) -> None:
        if action is None:
            return
        was_blocked = action.blockSignals(True)
        action.setChecked(bool(checked))
        action.blockSignals(was_blocked)

    def _set_soundboard_enabled(self, enabled: bool, persist: bool = True, silent: bool = False) -> bool:
        active = bool(self.engine.set_soundboard_enabled(enabled))
        self._set_action_checked(self._tray_soundboard_power_action, active)
        self._app_state.soundboardEnabled = active
        if persist:
            save_app_state(self._app_state)
        if not active:
            self._speaker_player.stop()
        if not silent:
            self.status_label.setText("Soundboard enabled." if active else "Soundboard disabled (mic only).")
        return active

    def _sync_startup_tray_state(self) -> None:
        startup_enabled = bool(self._app_state.startupEnabled)
        self._set_action_checked(self._tray_startup_action, startup_enabled)
        if self._tray_start_soundboard_action is not None:
            self._tray_start_soundboard_action.setVisible(startup_enabled)
            self._tray_start_soundboard_action.setEnabled(startup_enabled)
            self._set_action_checked(
                self._tray_start_soundboard_action,
                bool(self._app_state.startSoundboardOnLaunch),
            )
        self._set_action_checked(self._tray_notifications_action, bool(self._app_state.allowNotifications))
        self._set_action_checked(self._tray_soundboard_power_action, self.engine.is_soundboard_enabled())

    def _toggle_start_with_windows(self, checked: bool) -> None:
        target = bool(checked)
        if target and not self._app_state.startupEnabled:
            reply = QMessageBox.question(
                self,
                "Start With Windows",
                "Allow SoundboardEZ to launch automatically when Windows starts?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._set_action_checked(self._tray_startup_action, False)
                self._sync_startup_tray_state()
                return

        ok, err = set_startup_enabled(target)
        if not ok:
            self._set_action_checked(self._tray_startup_action, self._app_state.startupEnabled)
            self.status_label.setText(f"Startup update failed: {err}")
            self._notify_desktop(self.status_label.text(), title="Startup", error=True)
            self._sync_startup_tray_state()
            return

        self._app_state.startupEnabled = target
        save_app_state(self._app_state)
        self.status_label.setText("Start with Windows enabled." if target else "Start with Windows disabled.")
        self._notify_desktop(self.status_label.text(), title="Startup")
        self._sync_startup_tray_state()

    def _toggle_start_soundboard_on_launch(self, checked: bool) -> None:
        self._app_state.startSoundboardOnLaunch = bool(checked)
        save_app_state(self._app_state)
        self.status_label.setText(
            "Startup soundboard mode: on." if checked else "Startup soundboard mode: off."
        )

    def _toggle_allow_notifications(self, checked: bool) -> None:
        self._app_state.allowNotifications = bool(checked)
        save_app_state(self._app_state)
        self.status_label.setText("Desktop notifications enabled." if checked else "Desktop notifications disabled.")

    def _toggle_soundboard_power(self, checked: bool) -> None:
        self._set_soundboard_enabled(bool(checked), persist=True, silent=False)

    def _restore_from_tray(self) -> None:
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    def _hide_to_tray(self, show_notice: bool = False) -> None:
        self.hide()
        if show_notice and not self._tray_hide_hint_shown:
            self._tray_hide_hint_shown = True
            self._notify_desktop("SoundboardEZ is still running in the system tray.")

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._restore_from_tray()

    def _quit_from_tray(self) -> None:
        self._quitting = True
        self.close()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _on_title_close_clicked(self) -> None:
        modifiers = QApplication.keyboardModifiers()
        if bool(modifiers & Qt.KeyboardModifier.ShiftModifier):
            self._quit_from_tray()
            return
        self.close()

    def _setup_desktop_notifications(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray_icon = None
            return

        try:
            if self._tray_icon is not None:
                return
            tray = QSystemTrayIcon(self)
            icon = self.windowIcon()
            if icon.isNull():
                style = self.style()
                if style is not None:
                    icon = style.standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
            tray.setIcon(icon)
            tray.setToolTip("SoundboardEZ")
            tray.activated.connect(self._on_tray_activated)

            menu = QMenu(self)
            open_action = QAction("Open App", self)
            open_action.triggered.connect(self._restore_from_tray)
            menu.addAction(open_action)

            menu.addSeparator()

            startup_action = QAction("Start With Windows", self)
            startup_action.setCheckable(True)
            startup_action.toggled.connect(self._toggle_start_with_windows)
            menu.addAction(startup_action)

            startup_soundboard_action = QAction("Start Soundboard On Launch", self)
            startup_soundboard_action.setCheckable(True)
            startup_soundboard_action.toggled.connect(self._toggle_start_soundboard_on_launch)
            menu.addAction(startup_soundboard_action)

            notifications_action = QAction("Allow Notifications", self)
            notifications_action.setCheckable(True)
            notifications_action.toggled.connect(self._toggle_allow_notifications)
            menu.addAction(notifications_action)

            menu.addSeparator()

            soundboard_power_action = QAction("Soundboard Enabled", self)
            soundboard_power_action.setCheckable(True)
            soundboard_power_action.toggled.connect(self._toggle_soundboard_power)
            menu.addAction(soundboard_power_action)

            menu.addSeparator()
            quit_action = QAction("Quit", self)
            quit_action.triggered.connect(self._quit_from_tray)
            menu.addAction(quit_action)

            tray.setContextMenu(menu)
            tray.setVisible(True)
            self._tray_icon = tray
            self._tray_menu = menu
            self._tray_open_action = open_action
            self._tray_startup_action = startup_action
            self._tray_start_soundboard_action = startup_soundboard_action
            self._tray_notifications_action = notifications_action
            self._tray_soundboard_power_action = soundboard_power_action
            self._tray_quit_action = quit_action
            self._sync_startup_tray_state()
        except Exception:
            self._tray_icon = None

    def _notify_desktop(self, message: str, title: str = "SoundboardEZ", error: bool = False) -> None:
        if not bool(self._app_state.allowNotifications):
            return
        text = " ".join(str(message).split()).strip()
        if not text:
            return
        tray = self._tray_icon
        if tray is None or not tray.isVisible():
            return
        icon = QSystemTrayIcon.MessageIcon.Critical if error else QSystemTrayIcon.MessageIcon.Information
        try:
            tray.showMessage(title, text, icon, 4200)
        except Exception:
            pass

    @staticmethod
    def _is_virtual_cable_output_name(name: str) -> bool:
        lowered = str(name).lower()
        return (
            "cable input" in lowered
            or "vb-audio virtual cable" in lowered
            or "virtual cable" in lowered
        )

    def _resolve_aux_output_device(self, force_refresh: bool = False) -> int | None:
        """Return the system's current default output device index.

        This is used for preview playback and Play-to-Speaker — it should
        always honour the OS-level default so the user hears audio on
        whichever device they have selected in Windows Sound settings.
        """
        now = time.monotonic()
        if not force_refresh and now < self._aux_output_cache_deadline:
            return self._aux_output_device_cache

        default_out: int | None = None
        try:
            defaults = sd.default.device          # _InputOutputPair
            candidate = int(defaults[1])           # output index
            if candidate >= 0:
                default_out = candidate
        except Exception:
            default_out = None

        if default_out is not None:
            try:
                dev = sd.query_devices(default_out)
                if int(dev.get("max_output_channels", 0)) > 0:
                    self._aux_output_device_cache = default_out
                    self._aux_output_cache_deadline = now + 6.0
                    return default_out
            except Exception:
                pass

        # Fallback: pick any real output device if no default is set.
        candidates: list[tuple[int, int, int]] = []
        try:
            devices = sd.query_devices()
        except Exception:
            devices = []
        for idx, dev in enumerate(devices):
            if int(dev.get("max_output_channels", 0)) <= 0:
                continue
            name = str(dev.get("name", ""))
            if not name or self._is_virtual_cable_output_name(name):
                continue
            host_name = self.engine._hostapi_name(int(dev.get("hostapi", 0)))
            host_priority = self.engine._hostapi_priority(host_name)
            candidates.append((host_priority, idx))

        if not candidates:
            self._aux_output_device_cache = None
            self._aux_output_cache_deadline = now + 2.0
            return None

        candidates.sort()
        chosen = int(candidates[0][1])
        self._aux_output_device_cache = chosen
        self._aux_output_cache_deadline = now + 6.0
        return chosen

    def _run_worker(self, worker: QObject, on_finished, on_error, on_progress=None) -> None:
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def safe_finished(*args):
            try:
                on_finished(*args)
            except Exception as exc:
                self.status_label.setText("Operation failed unexpectedly.")
                self._notify_desktop(str(exc), title="Worker Error", error=True)
                print(f"Worker finished handler error: {exc}")

        def safe_error(*args):
            try:
                on_error(*args)
            except Exception as exc:
                self.status_label.setText("Operation failed unexpectedly.")
                self._notify_desktop(str(exc), title="Worker Error", error=True)
                print(f"Worker error handler error: {exc}")

        def safe_progress(*args):
            if on_progress is None:
                return
            try:
                on_progress(*args)
            except Exception as exc:
                print(f"Worker progress handler error: {exc}")

        worker.finished.connect(safe_finished)
        worker.error.connect(safe_error)
        if on_progress is not None and hasattr(worker, "progress"):
            try:
                worker.progress.connect(safe_progress)  # type: ignore[attr-defined]
            except Exception:
                pass
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        def cleanup() -> None:
            if thread in self._threads:
                self._threads.remove(thread)
            if worker in self._workers:
                self._workers.remove(worker)
        thread.finished.connect(cleanup)
        self._threads.append(thread)
        self._workers.append(worker)
        thread.start()

    def _position_importer_window(self) -> None:
        frame = self.frameGeometry()
        if frame.isNull():
            return
        x = frame.x() + max(24, int(frame.width() * 0.12))
        y = frame.y() + max(20, int(frame.height() * 0.08))
        self.importer_window.move(x, y)

    def _set_importer_visible(self, visible: bool) -> None:
        if visible:
            if not self.importer_window.isVisible():
                self._position_importer_window()
            self.importer_window.show()
            self.importer_window.raise_()
            self.importer_window.activateWindow()
        else:
            self._import_search_timer.stop()
            self._feed_row_render_timer.stop()
            self._pending_feed_rows.clear()
            self.importer_window.hide()
        self.toggle_importer_btn.setText("Close Importer" if visible else "Open Importer")
        if visible and not self._importer_loaded_once:
            self.fetch_sounds()
            self._importer_loaded_once = True

    def toggle_importer_panel(self) -> None:
        self._set_importer_visible(not self.importer_window.isVisible())

    def close_importer_panel(self) -> None:
        self._import_search_timer.stop()
        self._set_importer_visible(False)

    def _schedule_import_search(self, _text: str) -> None:
        if not self.importer_window.isVisible():
            return
        self._import_search_timer.start()

    def _run_debounced_import_search(self) -> None:
        if not self.importer_window.isVisible():
            return
        self.apply_import_search()

    def apply_import_search(self) -> None:
        self._import_search_timer.stop()
        self.fetch_sounds()

    def _begin_import_progress(self, title: str, total_items: int = 0) -> None:
        if self._import_progress_dialog is not None:
            try:
                self._import_progress_dialog.close()
            except Exception:
                pass
            self._import_progress_dialog.deleteLater()
            self._import_progress_dialog = None

        dialog = QProgressDialog("Preparing import...", None, 0, 100, self)
        dialog.setWindowTitle(title)
        dialog.setCancelButton(None)
        # Use NonModal so the main window stays responsive and doesn't
        # trigger black-border / "not responding" artifacts on Windows.
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)
        dialog.show()
        self._import_progress_dialog = dialog

    def _end_import_progress(self) -> None:
        dialog = self._import_progress_dialog
        self._import_progress_dialog = None
        if dialog is None:
            return
        try:
            dialog.close()
        except Exception:
            pass
        dialog.deleteLater()

    def _on_import_progress(self, payload: object) -> None:
        dialog = self._import_progress_dialog
        if dialog is None or not isinstance(payload, dict):
            return
        phase = str(payload.get("phase", "")).strip().lower()
        item_name = str(payload.get("item_name", "")).strip()
        item_index = int(payload.get("item_index", 0) or 0)
        total_items = int(payload.get("total_items", 0) or 0)
        downloaded = payload.get("downloaded_bytes")
        total = payload.get("total_bytes")

        if phase in {"download", "copy"}:
            verb = "Importing" if phase == "copy" else "Downloading"
            if isinstance(total, int) and total > 0 and isinstance(downloaded, int):
                dialog.setRange(0, total)
                dialog.setValue(max(0, min(total, int(downloaded))))
                label = (
                    f"{verb} {item_name} ({item_index}/{max(1, total_items)}) "
                    f"{int(downloaded) / (1024 * 1024):.1f}/{int(total) / (1024 * 1024):.1f} MB"
                )
                dialog.setLabelText(label)
            else:
                dialog.setRange(0, 0)
                dialog.setLabelText(
                    f"{verb} {item_name} ({item_index}/{max(1, total_items)})..."
                )
            return

        if phase == "cache":
            dialog.setRange(0, max(1, total_items))
            dialog.setValue(max(0, min(int(item_index), max(1, total_items))))
            dialog.setLabelText(f"Loading {item_name} ({item_index}/{max(1, total_items)})...")
            return

        if phase == "item-start":
            dialog.setRange(0, 0)
            dialog.setLabelText(f"Preparing {item_name} ({item_index}/{max(1, total_items)})...")
            return

    def import_from_link(self) -> None:
        raw = self.import_search_input.text().strip()
        if not raw:
            self.status_label.setText("Paste a link first.")
            return
        if not raw.startswith(("http://", "https://")):
            self.status_label.setText("Import Link expects a URL.")
            return

        parsed = urlparse(raw)
        host = (parsed.netloc or "").lower()
        if "youtube.com" in host or "youtu.be" in host:
            self.status_label.setText("Importing YouTube (first 5 seconds)...")
            worker = YoutubeImportWorker(raw, self.sounds_dir, clip_seconds=5.0)
            self._begin_import_progress("Importing YouTube", total_items=1)
            self._run_worker(worker, self._on_import_done, self._on_import_error)
            return

        if "myinstants.com" in host:
            audio_url = raw
            if "/media/sounds/" not in parsed.path:
                self.status_label.setText("For myinstants link import, use direct media link or feed Import button.")
                return
            name = Path(parsed.path).stem or "myinstants_sound"
            worker = ImportWorker([RemoteSoundItem(name=name, url=audio_url)], self.sounds_dir)
            self._begin_import_progress("Importing Sound", total_items=1)
            self._run_worker(worker, self._on_import_done, self._on_import_error, on_progress=self._on_import_progress)
            return

        self.status_label.setText("Only YouTube and myinstants links are supported.")

    def import_from_file_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Audio Files",
            "",
            "Audio Files (*.mp3 *.wav *.ogg *.m4a *.aac *.flac);;All Files (*.*)",
        )
        if not files:
            return
        self.status_label.setText(f"Importing {len(files)} local file(s)...")
        worker = FileImportWorker(files, self.sounds_dir)
        self._begin_import_progress("Importing Files", total_items=len(files))
        self._run_worker(worker, self._on_import_done, self._on_import_error, on_progress=self._on_import_progress)

    def fetch_sounds(self) -> None:
        raw_input = self.import_search_input.text().strip()
        raw = raw_input or MYINSTANTS_INDEX_URL
        url, error = _resolve_myinstants_feed_url(raw)
        if error or url is None:
            self.status_label.setText(error or "Invalid URL.")
            return

        # Keep the importer field as a simple search box for plain-text queries.
        # Only normalize the UI text when the user entered a direct URL.
        if raw_input.startswith(("http://", "https://")) and raw_input != url:
            blocked = self.import_search_input.blockSignals(True)
            self.import_search_input.setText(url)
            self.import_search_input.blockSignals(blocked)

        self._feed_request_id += 1
        self._feed_url = url
        self._feed_page = 0
        self._feed_loading = False
        self._fetch_in_progress = False
        self._feed_end_reached = False
        self._fetch_timeout.stop()
        self._feed_row_render_timer.stop()
        self._pending_feed_rows.clear()
        self._remote_seen_urls.clear()
        self._feed_play_buttons.clear()
        self.remote_feed_list.clear()
        self._load_next_feed_page()

    def _load_next_feed_page(self) -> None:
        if self._feed_loading or self._feed_end_reached:
            return

        next_page = self._feed_page + 1
        notify_completion = next_page == 1
        request_id = int(self._feed_request_id)
        self._feed_loading = True
        self._fetch_in_progress = True
        self._fetch_timeout.start(20000)
        self.fetch_btn.setEnabled(False)
        self.status_label.setText(f"Loading sounds page {next_page}...")
        cache_key = (self._feed_url, next_page)
        worker = FetchWorker(self._feed_url, next_page)

        def done(rows: list[tuple[str, str]]) -> None:
            if request_id != self._feed_request_id:
                return
            self._fetch_in_progress = False
            self._feed_loading = False
            self._fetch_timeout.stop()
            self.fetch_btn.setEnabled(True)
            self._feed_page_cache[cache_key] = list(rows)
            if len(self._feed_page_cache) > 160:
                try:
                    oldest_key = next(iter(self._feed_page_cache))
                    if oldest_key != cache_key:
                        self._feed_page_cache.pop(oldest_key, None)
                except Exception:
                    pass
            if not rows:
                if next_page == 1:
                    self.status_label.setText("No sounds found on this page URL.")
                else:
                    self.status_label.setText("Reached end of available sounds.")
                self._feed_end_reached = True
                if notify_completion:
                    self._notify_desktop(self.status_label.text(), title="Feed Load")
                return

            new_rows: list[RemoteSoundItem] = []
            for name, url in rows:
                if url in self._remote_seen_urls:
                    continue
                self._remote_seen_urls.add(url)
                new_rows.append(RemoteSoundItem(name=name, url=url))
            added = len(new_rows)
            if added:
                self._pending_feed_rows.extend(new_rows)
                if not self._feed_row_render_timer.isActive():
                    self._feed_row_render_timer.start()

            self._feed_page = next_page
            if added == 0:
                self._feed_end_reached = True
                self.status_label.setText("No new sounds on next page.")
            else:
                self.status_label.setText(f"Loaded page {next_page} ({added} sounds). Scroll for more.")
            if notify_completion:
                self._notify_desktop(self.status_label.text(), title="Feed Load")

        def err(message: str) -> None:
            if request_id != self._feed_request_id:
                return
            self._fetch_in_progress = False
            self._feed_loading = False
            self._fetch_timeout.stop()
            self.fetch_btn.setEnabled(True)
            friendly = self._friendly_fetch_error(message)
            self.status_label.setText(friendly)
            if notify_completion:
                self._notify_desktop(friendly, title="Feed Load", error=True)
            print(f"Feed load error: {message}")

        cached_rows = self._feed_page_cache.get(cache_key)
        if cached_rows is not None:
            QTimer.singleShot(0, lambda rows=list(cached_rows): done(rows))
            return

        self._run_worker(worker, done, err)

    @staticmethod
    def _friendly_fetch_error(message: str) -> str:
        text = str(message or "").strip()
        lower = text.lower()
        if not text:
            return "Feed load failed. Please try again."
        if "connectionreseterror" in lower or "forcibly closed by the remote host" in lower:
            return "Connection was reset by myinstants. Try Reload Feed in a moment."
        if "timed out" in lower or "readtimeout" in lower or "connecttimeout" in lower:
            return "Feed request timed out. Check connection and retry."
        if "name or service not known" in lower or "failed to establish a new connection" in lower:
            return "Cannot reach myinstants right now. Check internet/VPN and retry."
        if "403" in lower or "forbidden" in lower:
            return "myinstants rejected this request. Try again shortly."
        if "404" in lower:
            return "Feed URL not found. Check the search URL."
        if "unable to load myinstants feed:" in lower:
            return text.split(":", 1)[0].strip() + ". Please retry."
        return "Feed load failed. Please retry."

    def _on_feed_scroll(self, value: int) -> None:
        self._pending_feed_scroll_value = int(value)
        if not self._feed_scroll_debounce.isActive():
            self._feed_scroll_debounce.start()

    def _load_next_feed_page_if_needed(self) -> None:
        bar = self.remote_feed_list.verticalScrollBar()
        if bar.maximum() <= 0:
            return
        if int(self._pending_feed_scroll_value) >= bar.maximum() - 80:
            self._load_next_feed_page()

    def _drain_pending_feed_rows(self) -> None:
        if not self._pending_feed_rows:
            return
        self.remote_feed_list.setUpdatesEnabled(False)
        try:
            count = min(int(self._feed_row_chunk_size), len(self._pending_feed_rows))
            for _ in range(count):
                self._add_feed_row(self._pending_feed_rows.popleft())
        finally:
            self.remote_feed_list.setUpdatesEnabled(True)
        if self._pending_feed_rows:
            self._feed_row_render_timer.start()

    def _add_feed_row(self, item: RemoteSoundItem) -> None:
        row_item = QListWidgetItem()
        row_widget = QWidget()
        row_widget.setObjectName("FeedRow")
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(10, 10, 10, 10)
        row_layout.setSpacing(10)
        row_widget.setMinimumHeight(56)

        name_lbl = QLabel(item.name)
        name_lbl.setObjectName("FeedNameLabel")
        name_lbl.setToolTip(item.url)
        play_btn = QPushButton("Play")
        play_btn.setObjectName("FeedButton")
        play_btn.setMinimumHeight(32)
        play_btn.setMinimumWidth(78)
        play_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        import_btn = QPushButton("Import")
        import_btn.setObjectName("FeedButton")
        import_btn.setMinimumHeight(32)
        import_btn.setMinimumWidth(102)
        import_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._style_feed_buttons(play_btn, import_btn, seed=item.url)
        play_btn.clicked.connect(lambda _=False, it=item: self.toggle_remote_play(it))
        import_btn.clicked.connect(lambda _=False, it=item: self.import_remote_item(it))
        self._feed_play_buttons[item.url] = play_btn

        row_layout.addWidget(name_lbl, 1)
        row_layout.addWidget(play_btn)
        row_layout.addWidget(import_btn)

        row_item.setSizeHint(QSize(0, 66))
        row_widget.mouseDoubleClickEvent = lambda _event, it=item: self.toggle_remote_play(it)  # type: ignore[attr-defined]
        self.remote_feed_list.addItem(row_item)
        self.remote_feed_list.setItemWidget(row_item, row_widget)

    def _style_feed_buttons(self, play_btn: QPushButton, import_btn: QPushButton, seed: str) -> None:
        _ = seed
        play_btn.setProperty("active", False)
        import_btn.setProperty("active", False)

    def toggle_remote_play(self, item: RemoteSoundItem) -> None:
        if self._current_preview_url == item.url:
            self.stop_remote_preview()
            return
        self.start_remote_preview(item)

    def start_remote_preview(self, item: RemoteSoundItem) -> None:
        self._speaker_player.stop()
        self.stop_remote_preview(silent=True)
        self._preview_request_id += 1
        request_id = self._preview_request_id
        self._set_feed_play_button(item.url, "Loading...", enabled=False)
        self.status_label.setText(f"Loading '{item.name}'...")
        worker = PreviewWorker(item, volume=self._volume_state.preview_gain)

        def done(payload: object) -> None:
            if request_id != self._preview_request_id:
                return
            data = payload  # type: ignore[assignment]
            try:
                remote_item = data["item"]
                audio = np.asarray(data["audio"], dtype=np.float32)
                sr = data["sr"]
                if audio.ndim == 1:
                    audio = audio.reshape(-1, 1)
                elif audio.ndim > 2:
                    audio = audio.reshape(audio.shape[0], -1)
                if not audio.flags["C_CONTIGUOUS"]:
                    audio = np.ascontiguousarray(audio, dtype=np.float32)
                if audio.shape[0] == 0 or audio.shape[1] == 0:
                    raise RuntimeError("Preview audio is empty.")
            except Exception as exc:
                self._reset_feed_play_buttons()
                self.status_label.setText("Preview failed.")
                QMessageBox.warning(self, "Preview Error", str(exc))
                return

            # Resolve device + open stream entirely off the UI thread.
            # sd.query_devices() and Pa_OpenStream can block for seconds.
            preview_item = remote_item
            preview_audio = audio
            preview_sr = sr

            def _start_preview_playback() -> None:
                try:
                    device = self._resolve_aux_output_device()
                    if device is None:
                        raise RuntimeError("No non-virtual speaker output device is available for preview.")
                    try:
                        self._preview_player.play(preview_audio, samplerate=preview_sr, device=device)
                    except Exception:
                        device = self._resolve_aux_output_device(force_refresh=True)
                        if device is None:
                            raise RuntimeError("No non-virtual speaker output device is available for preview.")
                        self._preview_player.play(preview_audio, samplerate=preview_sr, device=device)
                except Exception as exc:
                    # Schedule UI updates back on the main thread.
                    QTimer.singleShot(0, lambda: self._on_preview_play_failed(str(exc)))
                    return
                # Schedule UI updates back on the main thread.
                QTimer.singleShot(0, lambda: self._on_preview_play_started(preview_item))

            threading.Thread(target=_start_preview_playback, daemon=True).start()

        def err(message: str) -> None:
            if request_id != self._preview_request_id:
                return
            self._reset_feed_play_buttons()
            self.status_label.setText("Preview failed.")
            QMessageBox.warning(self, "Preview Error", message)

        self._run_worker(worker, done, err)

    def stop_remote_preview(self, silent: bool = False) -> None:
        self._preview_request_id += 1
        self._preview_player.stop()
        self._current_preview_url = None
        self._preview_monitor.stop()
        self._reset_feed_play_buttons()
        if not silent:
            self.status_label.setText("Preview stopped.")

    def _on_preview_play_started(self, remote_item: RemoteSoundItem) -> None:
        """Called on the UI thread after background preview playback begins."""
        self._current_preview_url = remote_item.url
        self._reset_feed_play_buttons()
        self._set_feed_play_button(remote_item.url, "Stop", enabled=True)
        self._preview_monitor.start()
        self.status_label.setText(f"Previewing '{remote_item.name}'.")

    def _on_preview_play_failed(self, message: str) -> None:
        """Called on the UI thread when background preview playback fails."""
        self._reset_feed_play_buttons()
        self.status_label.setText("Preview failed.")
        QMessageBox.warning(self, "Preview Error", message)

    def _set_feed_play_button(self, url: str, text: str, enabled: bool = True) -> None:
        btn = self._feed_play_buttons.get(url)
        if btn is not None:
            active = text.strip().lower() == "stop"
            changed = False
            if btn.text() != text:
                btn.setText(text)
                changed = True
            if btn.isEnabled() != bool(enabled):
                btn.setEnabled(bool(enabled))
                changed = True
            if bool(btn.property("active")) != active:
                btn.setProperty("active", active)
                changed = True
            if changed:
                self._refresh_dynamic_button_style(btn)

    def _reset_feed_play_buttons(self) -> None:
        for btn in self._feed_play_buttons.values():
            changed = False
            if btn.text() != "Play":
                btn.setText("Play")
                changed = True
            if not btn.isEnabled():
                btn.setEnabled(True)
                changed = True
            if bool(btn.property("active")):
                btn.setProperty("active", False)
                changed = True
            if changed:
                self._refresh_dynamic_button_style(btn)

    @staticmethod
    def _refresh_dynamic_button_style(button: QPushButton) -> None:
        style = button.style()
        if style is None:
            return
        style.unpolish(button)
        style.polish(button)
        button.update()

    def _check_preview_finished(self) -> None:
        if self._current_preview_url is None:
            self._preview_monitor.stop()
            return
        active = self._preview_player.is_active()
        if not active:
            self._current_preview_url = None
            self._preview_monitor.stop()
            self._reset_feed_play_buttons()
            self.status_label.setText("Preview finished.")

    def import_remote_item(self, item: RemoteSoundItem) -> None:
        # Queue imports instead of spawning a new worker per click.
        if not hasattr(self, "_import_queue"):
            self._import_queue: deque[RemoteSoundItem] = deque()
            self._import_running = False
        self._import_queue.append(item)
        if self._import_running:
            self.status_label.setText(f"Queued '{item.name}' for import...")
            return
        self._drain_import_queue()

    def _drain_import_queue(self) -> None:
        if not self._import_queue:
            self._import_running = False
            return
        item = self._import_queue.popleft()
        self._import_running = True
        self.stop_remote_preview(silent=True)
        self.status_label.setText(f"Importing '{item.name}'...")
        worker = ImportWorker([item], self.sounds_dir)
        self._begin_import_progress("Importing Sound", total_items=1)
        self._run_worker(worker, self._on_import_done_queued, self._on_import_error_queued, on_progress=self._on_import_progress)

    def _on_import_done_queued(self, result: dict) -> None:
        self._end_import_progress()
        imported = result.get("imported", [])
        skipped = result.get("skipped", [])
        self._finalize_import(imported, skipped)
        # Process next item in queue
        QTimer.singleShot(0, self._drain_import_queue)

    def _on_import_error_queued(self, message: str) -> None:
        self._end_import_progress()
        self.status_label.setText("Import failed.")
        self._notify_desktop(f"Import failed: {message}", title="Import", error=True)
        # Continue with next item in queue
        QTimer.singleShot(0, self._drain_import_queue)

    def _handle_fetch_timeout(self) -> None:
        if not self._fetch_in_progress:
            return
        self._fetch_in_progress = False
        self._feed_loading = False
        self.fetch_btn.setEnabled(True)
        self.status_label.setText("Fetch timed out. Try reload.")
        self._notify_desktop(self.status_label.text(), title="Feed Load", error=True)

    def _on_import_done(self, result: dict) -> None:
        self._end_import_progress()
        imported = result.get("imported", [])
        skipped = result.get("skipped", [])
        self._finalize_import(imported, skipped)

    def _on_import_error(self, message: str) -> None:
        self._end_import_progress()
        self.status_label.setText("Import failed.")
        self._notify_desktop(f"Import failed: {message}", title="Import", error=True)
        QMessageBox.critical(self, "Import Error", message)

    def _finalize_import(self, imported: list[str], skipped: list[str]) -> None:
        # Incremental append: add only the new files to the local list
        # instead of rebuilding the entire grid each time.
        if imported:
            self._append_imported_to_local(imported)
        self._queue_sound_load(imported)
        if imported:
            self.status_label.setText(f"Imported {len(imported)} sound(s).")
            self._notify_desktop(self.status_label.text(), title="Import")
        elif skipped:
            self.status_label.setText(skipped[0])
            self._notify_desktop(self.status_label.text(), title="Import", error=True)
        else:
            self.status_label.setText("Nothing imported.")
            self._notify_desktop(self.status_label.text(), title="Import")

    def _register_new_sounds(self, imported_paths: list[str]) -> None:
        self._queue_sound_load(imported_paths)

    def _queue_sound_load(self, imported_paths: list[str]) -> None:
        if not imported_paths:
            return
        for path_str in imported_paths:
            text = str(path_str).strip()
            if not text:
                continue
            if text not in self._pending_sound_loads:
                self._pending_sound_loads.append(text)
        if self._sound_load_in_progress:
            return
        self._start_next_sound_load_batch()

    def _start_next_sound_load_batch(self) -> None:
        if self._sound_load_in_progress or not self._pending_sound_loads:
            return
        batch = list(self._pending_sound_loads)
        self._pending_sound_loads.clear()
        worker = SoundLoadWorker(self.engine, batch)
        self._sound_load_in_progress = True

        def done(result: dict) -> None:
            self._sound_load_in_progress = False
            failed = result.get("failed", [])
            if failed:
                print(f"Sound cache load issues: {failed[0]}")
            if self._pending_sound_loads:
                self._start_next_sound_load_batch()

        def err(message: str) -> None:
            self._sound_load_in_progress = False
            print(f"Sound cache load error: {message}")
            if self._pending_sound_loads:
                self._start_next_sound_load_batch()

        self._run_worker(worker, done, err, on_progress=self._on_import_progress)

    def refresh_local(self) -> None:
        self.sounds_dir.mkdir(parents=True, exist_ok=True)
        self._local_all_items.clear()
        for ext in ("*.wav", "*.mp3"):
            for audio_path in sorted(self.sounds_dir.glob(ext)):
                self._local_all_items.append(audio_path.name)
        self._update_local_grid_size()
        self.apply_local_filter()

    def _append_imported_to_local(self, imported: list[str]) -> None:
        """Incrementally add newly-imported files to the local grid without
        clearing and rebuilding the entire list.  Falls back to a full
        ``refresh_local()`` when a search filter is active."""
        query = self.local_search_input.text().strip().lower()
        tile_w, tile_h, _ = self._compute_local_tile_metrics()
        added = 0
        self.local_list.setUpdatesEnabled(False)
        try:
            for path_str in imported:
                name = Path(path_str).name
                if name in self._local_all_items:
                    continue
                self._local_all_items.append(name)
                if query and query not in name.lower():
                    continue
                visual_idx = self.local_list.count()
                item = QListWidgetItem(name)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setSizeHint(QSize(tile_w, tile_h))
                self._style_local_tile(item, visual_idx)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.local_list.addItem(item)
                added += 1
        finally:
            self.local_list.setUpdatesEnabled(True)
        if added:
            self._update_local_grid_size()

    def apply_local_filter(self, _text: str | None = None) -> None:
        query = self.local_search_input.text().strip().lower()
        tile_w, tile_h, _ = self._compute_local_tile_metrics()
        self.local_list.setUpdatesEnabled(False)
        self.local_list.clear()
        visual_idx = 0
        try:
            for name in self._local_all_items:
                if query and query not in name.lower():
                    continue
                item = QListWidgetItem(name)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setSizeHint(QSize(tile_w, tile_h))
                self._style_local_tile(item, visual_idx)
                visual_idx += 1
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.local_list.addItem(item)
        finally:
            self.local_list.setUpdatesEnabled(True)
        self._last_local_tile_size = QSize(tile_w, tile_h)
        if self._delete_mode:
            self._set_delete_checkboxes(True)

    def _style_local_tile(self, item: QListWidgetItem, idx: int) -> None:
        key = item.text().strip().lower()
        colors = self._local_tile_colors.get(key)
        if colors is None:
            colors = self._build_random_tile_colors(key)
            self._local_tile_colors[key] = colors
        item.setData(int(Qt.ItemDataRole.UserRole), colors)
        item.setForeground(QBrush(QColor("#e2e8f0")))
        item.setToolTip(item.text())

    def _build_random_tile_colors(self, seed_text: str) -> tuple[str, str]:
        _ = seed_text
        return ("#334155", "#334155")

    def _animate_local_tile_click(self, item: QListWidgetItem) -> None:
        base_colors = item.data(int(Qt.ItemDataRole.UserRole))
        if not (isinstance(base_colors, tuple) and len(base_colors) == 2):
            return
        bright1 = (
            QColor(base_colors[0]).lighter(170).name(),
            QColor(base_colors[1]).lighter(170).name(),
        )
        bright2 = (
            QColor(base_colors[0]).lighter(135).name(),
            QColor(base_colors[1]).lighter(135).name(),
        )
        item.setData(int(Qt.ItemDataRole.UserRole), bright1)
        self.local_list.viewport().update()

        def pulse_mid() -> None:
            if item.listWidget() is self.local_list:
                item.setData(int(Qt.ItemDataRole.UserRole), bright2)
                self.local_list.viewport().update()

        def restore() -> None:
            if item.listWidget() is self.local_list:
                item.setData(int(Qt.ItemDataRole.UserRole), base_colors)
                self.local_list.viewport().update()

        QTimer.singleShot(110, pulse_mid)
        QTimer.singleShot(230, restore)

    def _compute_local_tile_metrics(self) -> tuple[int, int, int]:
        spacing = max(12, self.local_list.spacing())
        view_w = max(300, self.local_list.viewport().width())
        cols = 6
        tile_w = int((view_w - (spacing * (cols - 1))) / cols)
        min_tile_w = 100
        max_tile_w = 150
        tile_w = max(min_tile_w, min(max_tile_w, tile_w))
        tile_h = max(68, int(tile_w * 0.46))
        return tile_w, tile_h, spacing

    def _update_local_grid_size(self) -> None:
        tile_w, tile_h, spacing = self._compute_local_tile_metrics()
        grid_size = QSize(tile_w + spacing, tile_h + spacing)
        if self.local_list.gridSize() != grid_size:
            self.local_list.setGridSize(grid_size)
        visible_rows = 4
        min_height = (tile_h * visible_rows) + (spacing * (visible_rows - 1)) + 32
        if self.local_list.minimumHeight() != min_height:
            self.local_list.setMinimumHeight(min_height)
        self.local_list.setMinimumWidth(0)

    def _refresh_local_item_size_hints(self) -> None:
        tile_w, tile_h, _ = self._compute_local_tile_metrics()
        size = QSize(tile_w, tile_h)
        if self._last_local_tile_size == size:
            return
        self._last_local_tile_size = QSize(size)
        for idx in range(self.local_list.count()):
            item = self.local_list.item(idx)
            if item is not None:
                item.setSizeHint(size)

    def _apply_local_layout_refresh(self) -> None:
        self._update_local_grid_size()
        self._refresh_local_item_size_hints()

    def play_selected_imported(self) -> None:
        if self._delete_mode:
            self.status_label.setText("Exit delete mode to play sounds.")
            return
        item = self.local_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Play Imported", "Select an imported sound first.")
            return
        self.play_imported_item(item)

    def play_imported_item(self, item: QListWidgetItem) -> None:
        if self._delete_mode:
            self.status_label.setText("Exit delete mode to play sounds.")
            return
        if not self.engine.is_soundboard_enabled():
            self.status_label.setText("Soundboard is disabled (mic only).")
            return
        file_name = item.text().strip()
        file_path = self.sounds_dir / file_name
        if not file_path.exists():
            QMessageBox.warning(self, "Play Imported", f"File not found: {file_name}")
            self.refresh_local()
            return

        sound_name = file_path.stem
        try:
            self.engine.soundboard.stop_all()
            if sound_name in self.engine.soundboard.sounds:
                self.engine.soundboard.trigger(sound_name)
                self._play_sound_to_speaker_if_enabled(sound_name)
                self.status_label.setText(f"Playing '{sound_name}' to virtual mic.")
                return

            self.status_label.setText(f"Loading '{sound_name}'...")
            worker = SoundLoadWorker(self.engine, [str(file_path)])

            def done(result: dict) -> None:
                loaded = int(result.get("loaded", 0) or 0)
                if loaded <= 0:
                    failed = result.get("failed", [])
                    reason = failed[0] if isinstance(failed, list) and failed else "Unknown load error."
                    QMessageBox.critical(self, "Play Imported", f"Failed to load '{file_name}': {reason}")
                    return
                try:
                    self.engine.soundboard.stop_all()
                    self.engine.soundboard.trigger(sound_name)
                    self._play_sound_to_speaker_if_enabled(sound_name)
                    self.status_label.setText(f"Playing '{sound_name}' to virtual mic.")
                except Exception as exc:
                    QMessageBox.critical(self, "Play Imported", f"Failed to play '{file_name}': {exc}")

            def err(message: str) -> None:
                QMessageBox.critical(self, "Play Imported", f"Failed to load '{file_name}': {message}")

            self._run_worker(worker, done, err)
        except Exception as exc:
            QMessageBox.critical(self, "Play Imported", f"Failed to play '{file_name}': {exc}")

    def trim_selected_imported(self) -> None:
        if not self._trim_mode:
            self._enter_trim_mode()
            return
        current = self.local_list.currentItem()
        if current is None:
            self.status_label.setText("Select one sound to trim.")
            return
        self._open_trim_editor_for_item(current)

    def _on_local_item_clicked(self, item: QListWidgetItem) -> None:
        if self._delete_mode:
            return
        self._animate_local_tile_click(item)
        if self._trim_mode:
            self._open_trim_editor_for_item(item)

    def _show_local_context_menu(self, pos) -> None:
        item = self.local_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self.local_list)
        menu.setObjectName("LocalTileMenu")
        play_action = menu.addAction("Play")
        select_action = menu.addAction("Select")
        rename_action = menu.addAction("Rename")
        edit_action = menu.addAction("Edit (Trim)")
        delete_action = menu.addAction("Delete")
        menu.addSeparator()
        select_all_action = menu.addAction("Select All")
        delete_selected_action = menu.addAction("Delete Selected")

        action = menu.exec(self.local_list.viewport().mapToGlobal(pos))
        if action is None:
            return
        if action is play_action:
            self.play_imported_item(item)
        elif action is select_action:
            item.setCheckState(Qt.CheckState.Checked if item.checkState() != Qt.CheckState.Checked else Qt.CheckState.Unchecked)
        elif action is rename_action:
            self._rename_imported_item(item)
        elif action is edit_action:
            self._open_trim_editor_for_item(item)
        elif action is delete_action:
            self._delete_single_imported(item)
        elif action is select_all_action:
            for i in range(self.local_list.count()):
                it = self.local_list.item(i)
                it.setCheckState(Qt.CheckState.Checked)
        elif action is delete_selected_action:
            self.delete_selected_imported()

    def _enter_trim_mode(self) -> None:
        if self._delete_mode:
            self._exit_delete_mode()
        self._trim_mode = True
        self.trim_local_btn.setText("Pick Sound To Trim")
        self.cancel_trim_btn.setVisible(True)
        self.trim_mode_hint.setVisible(True)
        self.trim_dialog.hide()
        self._trim_source_path = None
        self._trim_audio = None
        self.status_label.setText("Trim mode enabled. Click a sound tile to open the trim window.")

    def _exit_trim_mode(self) -> None:
        self._trim_mode = False
        self.trim_local_btn.setText("Trim")
        self.cancel_trim_btn.setVisible(False)
        self.trim_mode_hint.setVisible(False)
        self.close_trim_editor()

    def cancel_trim_mode(self) -> None:
        if not self._trim_mode:
            return
        self._exit_trim_mode()
        self.status_label.setText("Trim canceled.")

    def _set_trim_editor_audio(self, src: Path, audio: np.ndarray, sr: int, status_text: str) -> None:
        self.stop_trim_preview(silent=True)
        self._trim_source_path = src
        self._trim_audio = np.ascontiguousarray(audio, dtype=np.float32)
        self._trim_sr = max(1, int(sr))
        self._trim_playhead = 0
        total_frames = max(1, int(self._trim_audio.shape[0]))
        max_ms = max(1, int((total_frames / self._trim_sr) * 1000))
        self._trim_updating_slider = True
        self.trim_dialog.timeline.set_duration_ms(max_ms)
        self.trim_dialog.timeline.set_range_ms(0, max_ms, emit=False)
        self.trim_dialog.timeline.set_playhead_ms(0, emit=False)
        self._trim_updating_slider = False
        self.trim_dialog.trim_target_label.setText(src.name)
        self.trim_dialog.trim_time_label.setText(f"00:00.00 / {self._format_ms(max_ms)}")
        self.trim_play_pause_btn.setText("Play")
        if not self.trim_dialog.isVisible():
            dialog_geo = self.trim_dialog.frameGeometry()
            dialog_geo.moveCenter(self.frameGeometry().center())
            self.trim_dialog.move(dialog_geo.topLeft())
            self.trim_dialog.show()
        self.trim_dialog.raise_()
        self.trim_dialog.activateWindow()
        self.status_label.setText(status_text)

    def _open_trim_editor_for_item(self, item: QListWidgetItem) -> None:
        file_name = item.text().strip()
        src = self.sounds_dir / file_name
        if not src.exists():
            self.status_label.setText("Selected file does not exist.")
            self.refresh_local()
            return
        self._trim_decode_request_id += 1
        request_id = int(self._trim_decode_request_id)
        self.status_label.setText(f"Loading trim audio for '{file_name}'...")
        worker = AudioDecodeWorker(src)

        def done(result: dict) -> None:
            if request_id != self._trim_decode_request_id:
                return
            audio = np.asarray(result.get("audio"), dtype=np.float32)
            sr = int(result.get("sr", 48000) or 48000)
            self._set_trim_editor_audio(src, audio, sr, f"Trim editor opened for '{file_name}'.")

        def err(message: str) -> None:
            if request_id != self._trim_decode_request_id:
                return
            self.status_label.setText(f"Failed to load trim audio: {message}")

        self._run_worker(worker, done, err)

    def _rename_imported_item(self, item: QListWidgetItem) -> None:
        old_name = item.text().strip()
        old_path = self.sounds_dir / old_name
        if not old_path.exists():
            self.status_label.setText("File not found.")
            return
        new_name, ok = QInputDialog.getText(self, "Rename Sound", "New name:", text=old_path.stem)
        if not ok:
            return
        new_name = "".join(ch for ch in new_name if ch not in "/\\?*:|\"<>").strip()
        if not new_name:
            self.status_label.setText("Name cannot be empty.")
            return
        new_path = old_path.with_name(f"{new_name}{old_path.suffix}")
        if new_path.exists():
            self.status_label.setText("A sound with that name already exists.")
            return
        try:
            old_path.rename(new_path)
            item.setText(new_path.name)
            self.status_label.setText(f"Renamed to '{new_path.name}'.")
        except Exception as exc:
            self.status_label.setText(f"Rename failed: {exc}")

    def _delete_single_imported(self, item: QListWidgetItem) -> None:
        file_name = item.text().strip()
        file_path = self.sounds_dir / file_name
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as exc:
            self.status_label.setText(f"Failed to delete '{file_name}': {exc}")
            return
        row = self.local_list.row(item)
        self.local_list.takeItem(row)
        self.engine.soundboard.sounds.pop(file_path.stem, None)
        self.status_label.setText(f"Deleted '{file_name}'.")

    def close_trim_editor(self) -> None:
        self.stop_trim_preview(silent=True)
        self.trim_dialog.hide()
        self._trim_source_path = None
        self._trim_audio = None

    def apply_trim_from_editor(self) -> None:
        src = self._trim_source_path
        if src is None or not src.exists():
            self.status_label.setText("No trim target selected.")
            return

        ffmpeg_exe = _get_ffmpeg_exe()
        if ffmpeg_exe is None:
            self.status_label.setText("ffmpeg is required for trim.")
            return

        start_ms = self.trim_dialog.timeline.start_ms()
        end_ms = self.trim_dialog.timeline.end_ms()
        if end_ms <= start_ms:
            self.status_label.setText("Trim end must be greater than trim start.")
            return
        start = start_ms / 1000.0
        length = (end_ms - start_ms) / 1000.0

        self.stop_trim_preview(silent=True, reset_to_start=False)
        self.apply_trim_btn.setEnabled(False)
        self.status_label.setText("Applying trim...")

        trim_worker = TrimApplyWorker(
            ffmpeg_exe=ffmpeg_exe,
            src=src,
            start=start,
            length=length,
            engine=self.engine,
        )

        self._trim_decode_request_id += 1
        request_id = int(self._trim_decode_request_id)

        def trim_done(result: dict) -> None:
            self.apply_trim_btn.setEnabled(True)
            reload_error = result.get("reload_error")
            if reload_error:
                self.status_label.setText(f"Trim saved, but failed to reload sound: {reload_error}")
                self._notify_desktop(self.status_label.text(), title="Trim", error=True)
                self.refresh_local()
                return

            self.refresh_local()
            for i in range(self.local_list.count()):
                item = self.local_list.item(i)
                if item.text().strip().lower() == src.name.lower():
                    self.local_list.setCurrentItem(item)
                    break

            self.status_label.setText("Trim saved. Reloading editor...")
            decode_worker = AudioDecodeWorker(src)

            def decode_done(decode_result: dict) -> None:
                if request_id != self._trim_decode_request_id:
                    return
                audio = np.asarray(decode_result.get("audio"), dtype=np.float32)
                sr = int(decode_result.get("sr", 48000) or 48000)
                self._set_trim_editor_audio(src, audio, sr, f"Trim saved to '{src.name}'.")
                self._notify_desktop(self.status_label.text(), title="Trim")

            def decode_err(message: str) -> None:
                if request_id != self._trim_decode_request_id:
                    return
                self.status_label.setText(f"Trim saved, but failed to reopen editor: {message}")
                self._notify_desktop(self.status_label.text(), title="Trim", error=True)

            self._run_worker(decode_worker, decode_done, decode_err)

        def trim_err(message: str) -> None:
            self.apply_trim_btn.setEnabled(True)
            self.status_label.setText(f"Failed to trim: {message}")
            self._notify_desktop(self.status_label.text(), title="Trim", error=True)

        self._run_worker(trim_worker, trim_done, trim_err)

    def delete_selected_imported(self) -> None:
        if not self._delete_mode:
            self._enter_delete_mode()
            return

        names: list[str] = []
        for i in range(self.local_list.count()):
            item = self.local_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                names.append(item.text().strip())

        if not names:
            self.status_label.setText("No checked sounds to delete.")
            return

        deleted = 0
        for file_name in names:
            file_path = self.sounds_dir / file_name
            try:
                if file_path.exists():
                    file_path.unlink()
                    deleted += 1
            except Exception as exc:
                self.status_label.setText(f"Failed to delete '{file_name}': {exc}")
                continue

            sound_name = file_path.stem
            self.engine.soundboard.sounds.pop(sound_name, None)

        self._exit_delete_mode()
        self.refresh_local()
        self.status_label.setText(f"Deleted {deleted} sound(s).")
        self._notify_desktop(self.status_label.text(), title="Delete")

    def _enter_delete_mode(self) -> None:
        if self._trim_mode:
            self._exit_trim_mode()
        self._delete_mode = True
        self.delete_local_btn.setText("Delete Checked")
        self.cancel_delete_btn.setVisible(True)
        self.local_list.clearSelection()
        self.local_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._set_delete_checkboxes(True)
        self.delete_mode_hint.setVisible(True)
        self._delete_hint_fx.setOpacity(0.0)
        self._delete_hint_anim.start()
        self.status_label.setText("Delete mode enabled. Tick sounds, then click Delete Checked.")

    def _exit_delete_mode(self) -> None:
        self._delete_mode = False
        self.delete_local_btn.setText("Delete Selected")
        self.cancel_delete_btn.setVisible(False)
        self.delete_mode_hint.setVisible(False)
        self._set_delete_checkboxes(False)
        self.local_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

    def cancel_delete_mode(self) -> None:
        if not self._delete_mode:
            return
        self._exit_delete_mode()
        self.status_label.setText("Delete canceled.")

    def _set_delete_checkboxes(self, enabled: bool) -> None:
        for i in range(self.local_list.count()):
            item = self.local_list.item(i)
            if enabled:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
            else:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)

    def _on_trim_range_changed(self, _value: int) -> None:
        if self._trim_updating_slider:
            return
        timeline = self.trim_dialog.timeline
        start = timeline.start_ms()
        end = timeline.end_ms()
        if self._trim_playhead < start:
            self._trim_playhead = start
            self._trim_updating_slider = True
            timeline.set_playhead_ms(start, emit=False)
            self._trim_updating_slider = False
        if self._trim_playhead > end:
            self._trim_playhead = end
            self._trim_updating_slider = True
            timeline.set_playhead_ms(end, emit=False)
            self._trim_updating_slider = False
        max_ms = max(timeline.duration_ms(), 1)
        self.trim_dialog.trim_time_label.setText(f"{self._format_ms(self._trim_playhead)} / {self._format_ms(max_ms)}")

    def _on_trim_timeline_changed(self, value: int) -> None:
        if self._trim_updating_slider:
            return
        self._trim_playhead = value
        max_ms = max(self.trim_dialog.timeline.duration_ms(), 1)
        self.trim_dialog.trim_time_label.setText(f"{self._format_ms(value)} / {self._format_ms(max_ms)}")

    def _sync_trim_playhead(self) -> None:
        if not self._trim_playing:
            return
        self._trim_updating_slider = True
        self.trim_dialog.timeline.set_playhead_ms(self._trim_playhead, emit=False)
        self._trim_updating_slider = False
        max_ms = max(self.trim_dialog.timeline.duration_ms(), 1)
        self.trim_dialog.trim_time_label.setText(f"{self._format_ms(self._trim_playhead)} / {self._format_ms(max_ms)}")
        if self._trim_playhead >= self.trim_dialog.timeline.end_ms():
            self.stop_trim_preview(silent=True, reset_to_start=True)

    def toggle_trim_preview_play(self) -> None:
        if self._trim_audio is None:
            self.status_label.setText("Open trim editor for a sound first.")
            return
        if self._trim_playing:
            self.pause_trim_preview()
            return
        self.start_trim_preview()

    def start_trim_preview(self) -> None:
        if self._trim_audio is None:
            return
        range_start_ms = self.trim_dialog.timeline.start_ms()
        start_ms = max(range_start_ms, self._trim_playhead)
        end_ms = self.trim_dialog.timeline.end_ms()
        if start_ms >= end_ms:
            start_ms = range_start_ms
        if end_ms <= start_ms:
            self.status_label.setText("Invalid trim range.")
            return

        start_frame = int((start_ms / 1000.0) * self._trim_sr)
        end_frame = int((end_ms / 1000.0) * self._trim_sr)
        self._trim_playhead = start_ms
        self._trim_updating_slider = True
        self.trim_dialog.timeline.set_playhead_ms(self._trim_playhead, emit=False)
        self._trim_updating_slider = False
        self.stop_trim_preview(silent=True, reset_to_start=False)

        play_pos = {"frame": start_frame}

        def callback(outdata, frames, _time, _status):
            cur = play_pos["frame"]
            remaining = end_frame - cur
            if remaining <= 0:
                outdata.fill(0)
                self._trim_playing = False
                raise sd.CallbackStop
            n = min(frames, remaining)
            chunk = self._trim_audio[cur : cur + n]
            outdata.fill(0)
            outdata[:n, : chunk.shape[1]] = chunk * self._trim_preview_gain
            play_pos["frame"] = cur + n
            self._trim_playhead = int((play_pos["frame"] / self._trim_sr) * 1000)
            if n < frames:
                self._trim_playing = False
                raise sd.CallbackStop

        try:
            self._trim_stream = sd.OutputStream(
                samplerate=self._trim_sr,
                channels=self._trim_audio.shape[1],
                dtype="float32",
                callback=callback,
            )
            self._trim_stream.start()
            self._trim_playing = True
            self.trim_play_pause_btn.setText("Pause")
            self._trim_play_timer.start()
            self.status_label.setText("Trim preview playing.")
        except Exception as exc:
            self._trim_playing = False
            self._trim_stream = None
            self.status_label.setText(f"Trim preview failed: {exc}")

    def pause_trim_preview(self) -> None:
        if not self._trim_playing:
            return
        self.stop_trim_preview(silent=True, reset_to_start=False)
        self.trim_play_pause_btn.setText("Play")
        self.status_label.setText("Trim preview paused.")

    def stop_trim_preview(self, silent: bool = False, reset_to_start: bool = False) -> None:
        self._trim_playing = False
        self._trim_play_timer.stop()
        if self._trim_stream is not None:
            try:
                self._trim_stream.stop()
                self._trim_stream.close()
            except Exception:
                pass
            self._trim_stream = None
        if reset_to_start:
            self._trim_playhead = self.trim_dialog.timeline.start_ms()
            self._trim_updating_slider = True
            self.trim_dialog.timeline.set_playhead_ms(self._trim_playhead, emit=False)
            self._trim_updating_slider = False
            max_ms = max(self.trim_dialog.timeline.duration_ms(), 1)
            self.trim_dialog.trim_time_label.setText(f"{self._format_ms(self._trim_playhead)} / {self._format_ms(max_ms)}")
        self.trim_play_pause_btn.setText("Play")
        if not silent:
            self.status_label.setText("Trim preview stopped.")

    def _format_ms(self, ms: int) -> str:
        total = max(0, int(ms))
        s = total / 1000.0
        minutes = int(s // 60)
        seconds = s - (minutes * 60)
        return f"{minutes:02d}:{seconds:05.2f}"

    def update_soundboard_volume(self, slider_value: int) -> None:
        gain = max(0.0, min(2.0, slider_value / 100.0))
        self._volume_state.soundboard_gain = gain
        self.engine.soundboard.set_volume(gain)
        self.soundboard_volume_label.setText(f"Soundboard Volume: {slider_value}%")

    def _sync_mic_noise_suppression_button(self) -> None:
        available = self.engine.is_noise_suppression_available()
        active = self.engine.is_noise_suppression_enabled() if available else False

        was_blocked = self.mic_noise_suppression_btn.blockSignals(True)
        self.mic_noise_suppression_btn.setChecked(active)
        self.mic_noise_suppression_btn.blockSignals(was_blocked)

        if not available:
            self.mic_noise_suppression_btn.setEnabled(False)
            self.mic_noise_suppression_btn.setText("Mic Noise Suppression: Unavailable")
            err = self.engine.noise_suppression_error()
            self.mic_noise_suppression_btn.setToolTip(err or "RNNoise backend unavailable.")
            return

        self.mic_noise_suppression_btn.setEnabled(True)
        self.mic_noise_suppression_btn.setText("Mic Noise Suppression: On" if active else "Mic Noise Suppression: Off")
        backend = self.engine.noise_suppression_backend()
        self.mic_noise_suppression_btn.setToolTip(f"RNNoise backend: {backend}")

    def toggle_mic_noise_suppression(self, enabled: bool) -> None:
        active = self.engine.set_noise_suppression_enabled(enabled)
        self._sync_mic_noise_suppression_button()
        if enabled and not active:
            err = self.engine.noise_suppression_error()
            self.status_label.setText(err or "Mic noise suppression unavailable.")
            return
        self.status_label.setText(
            "Mic noise suppression enabled." if active else "Mic noise suppression disabled."
        )

    def toggle_speaker_monitor(self, enabled: bool) -> None:
        self.speaker_monitor_btn.setText("Play To Speaker: On" if enabled else "Play To Speaker: Off")
        self.speaker_monitor_label.setVisible(enabled)
        self.speaker_monitor_slider.setVisible(enabled)
        if not enabled:
            self._speaker_player.stop()

    def update_speaker_monitor_volume_label(self, slider_value: int) -> None:
        gain = max(0.0, min(2.0, slider_value / 100.0))
        self._volume_state.speaker_gain = gain
        self._speaker_player.set_gain(gain)
        self.speaker_monitor_label.setText(f"Speaker Volume: {slider_value}%")

    def _play_sound_to_speaker_if_enabled(self, sound_name: str) -> None:
        if not self.engine.is_soundboard_enabled():
            return
        if not self.speaker_monitor_btn.isChecked():
            return
        self.stop_remote_preview(silent=True)
        snd = self.engine.soundboard.sounds.get(sound_name)
        if snd is None:
            return
        gain = float(self._volume_state.speaker_gain)
        if gain <= 0.0:
            return

        # Resolve device + open stream off the UI thread.  sd.query_devices()
        # and sd.OutputStream() can block for several seconds on Windows.
        audio_copy = np.array(snd.audio, dtype=np.float32, copy=True, order="C")
        sr = int(self.engine.samplerate)

        def _speaker_task() -> None:
            try:
                device = self._resolve_aux_output_device()
                if device is None:
                    return
                try:
                    self._speaker_player.play(audio_copy, samplerate=sr, device=device)
                except Exception:
                    device = self._resolve_aux_output_device(force_refresh=True)
                    if device is None:
                        return
                    self._speaker_player.play(audio_copy, samplerate=sr, device=device)
            except Exception as exc:
                print(f"Speaker playback failed: {exc}")

        threading.Thread(target=_speaker_task, daemon=True).start()

    def update_preview_volume_label(self, slider_value: int) -> None:
        gain = max(0.0, min(2.0, slider_value / 100.0))
        self._volume_state.preview_gain = gain
        self._preview_player.set_gain(gain)
        self.preview_volume_label.setText(f"Preview Volume: {slider_value}%")

    def update_mic_volume_label(self, slider_value: int) -> None:
        value = max(0, min(200, int(slider_value)))
        self.mic_volume_label.setText(f"Mic Volume: {value}%")
        gain = value / 100.0
        self._volume_state.mic_gain = gain
        self.engine.set_mic_input_gain(gain)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if not getattr(self, "_quitting", False) and self._tray_icon is not None and self._tray_icon.isVisible():
            event.ignore()
            self._hide_to_tray(show_notice=True)
            return

        self._persist_app_state()
        self._end_import_progress()
        self._fetch_timeout.stop()
        self._feed_scroll_debounce.stop()
        self._feed_row_render_timer.stop()
        self._import_search_timer.stop()
        self._local_layout_timer.stop()
        self._preview_monitor.stop()
        self._trim_play_timer.stop()
        self.stop_remote_preview(silent=True)
        self._speaker_player.close()
        self.stop_trim_preview(silent=True)
        self.engine.shutdown()
        if self._engine_thread.is_alive():
            self._engine_thread.join(timeout=2.5)
        for thread in list(self._threads):
            try:
                thread.quit()
                thread.wait(500)
            except Exception:
                pass
        if self._tray_icon is not None:
            try:
                self._tray_icon.hide()
            except Exception:
                pass
        super().closeEvent(event)

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if getattr(self, "_quitting", False):
            return
        if event.type() != QEvent.Type.WindowStateChange:
            return
        if self.isMinimized() and self._tray_icon is not None and self._tray_icon.isVisible():
            QTimer.singleShot(0, lambda: self._hide_to_tray(show_notice=False))

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_window_mask()
        self._local_layout_timer.start()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _should_check_updates(state: AppState) -> bool:
    return bool(state.autoUpdateEnabled)


def _run_startup_auto_update(
    app: QApplication,
    state: AppState,
    skip_update_once: bool,
) -> bool:
    if not getattr(sys, "frozen", False):
        # Keep local source runs fast; updater is intended for installed builds.
        return False

    optional_updates_enabled = _should_check_updates(state)

    update: UpdateInfo | None = None
    check_ok = False
    for attempt in range(1, 4):
        state.lastUpdateAttemptUtc = _utc_now_iso()
        try:
            update = check_for_update(APP_VERSION)
            check_ok = True
            break
        except Exception as exc:
            print(f"Startup update check failed (attempt {attempt}/3): {exc}")

    if check_ok:
        state.lastUpdateCheckUtc = _utc_now_iso()
        if update is not None:
            state.lastUpdateVersionSeen = str(update.version)
        save_app_state(state)
    else:
        save_app_state(state)
        return False

    if update is None:
        return False

    mandatory_update = bool(getattr(update, "mandatory", False))
    if not mandatory_update:
        if skip_update_once:
            return False
        if not optional_updates_enabled:
            return False

    update_info = update
    update_dir = Path(tempfile.gettempdir())
    update_dialog = UpdateDialog(APP_VERSION, update_info.version, mandatory=mandatory_update)
    update_running = {"value": False}
    worker_ref: dict[str, QObject | None] = {"worker": None}
    thread_ref: dict[str, QThread | None] = {"thread": None}

    def begin_update() -> None:
        if update_running["value"]:
            return
        update_running["value"] = True
        update_dialog.begin_update()

        worker = StartupUpdateWorker(update_info, update_dir)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.stage.connect(
            lambda stage: update_dialog.set_extracting()
            if str(stage).strip().lower() == "extract"
            else None
        )
        worker.progress.connect(update_dialog.set_download_progress)

        def done(payload: dict) -> None:
            if not isinstance(payload, dict):
                update_dialog.set_error("Update worker returned invalid payload.")
                return

            mode = str(payload.get("mode", "")).strip()
            temp_dir = str(payload.get("temp_dir", "")).strip()
            if not mode or not temp_dir:
                update_dialog.set_error("Update payload missing mode or temp_dir.")
                return

            if not Path(temp_dir).exists():
                update_dialog.set_error(f"Update temp directory missing: {temp_dir}")
                return

            update_dialog.set_installing()
            try:
                launch_apply_and_exit(mode, temp_dir, os.getpid())
            except SystemExit:
                update_dialog.finish_success()
                update_dialog.accept()
            except Exception as exc:
                update_dialog.set_error(f"Update handoff failed: {exc}")

        def err(message: str) -> None:
            update_dialog.set_error(str(message))

        worker.finished.connect(done)
        worker.error.connect(err)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        def cleanup() -> None:
            update_running["value"] = False
            worker_ref["worker"] = None
            thread_ref["thread"] = None

        thread.finished.connect(cleanup)
        worker_ref["worker"] = worker
        thread_ref["thread"] = thread
        thread.start()

    update_dialog.updateRequested.connect(begin_update)
    if mandatory_update:
        QTimer.singleShot(0, begin_update)
    dialog_code = update_dialog.exec()
    return dialog_code == QDialog.DialogCode.Accepted


def run_ui() -> int:
    launched_from_startup = STARTUP_ARG in sys.argv
    skip_update_once = SKIP_UPDATE_ONCE_ARG in sys.argv
    argv = [arg for arg in sys.argv if arg not in {STARTUP_ARG, SKIP_UPDATE_ONCE_ARG}]
    app = QApplication(argv)
    app.setQuitOnLastWindowClosed(False)
    app.setFont(QFont("Segoe UI", 11))

    lock_path = Path(tempfile.gettempdir()) / "SoundboardEZ.lock"
    instance_lock = QLockFile(str(lock_path))
    instance_lock.setStaleLockTime(0)
    if not instance_lock.tryLock(100):
        if not launched_from_startup:
            QMessageBox.information(None, "SoundboardEZ", "SoundboardEZ is already running.")
        return 0

    app_state = load_app_state()
    if _run_startup_auto_update(app, app_state, skip_update_once):
        try:
            instance_lock.unlock()
        except Exception:
            pass
        return 0

    app.setProperty("_single_instance_lock", instance_lock)
    window = SoundboardWindow(launched_from_startup=launched_from_startup)
    window.show()
    exit_code = app.exec()
    try:
        instance_lock.unlock()
    except Exception:
        pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run_ui())
