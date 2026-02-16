from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import threading
from urllib.parse import parse_qs, quote_plus, urlparse
import wave

import numpy as np
import requests
import sounddevice as sd
from PyQt6.QtCore import (
    QEvent,
    QEasingCurve,
    QObject,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    QPropertyAnimation,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QBrush, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from audio_engine import AudioEngine
from scraper import MYINSTANTS_INDEX_URL, fetch_myinstants_sounds_page

PHI = 1.618


@dataclass(frozen=True)
class RemoteSoundItem:
    name: str
    url: str


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


def _download(url: str, timeout: float = 20.0) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


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
            pcm = np.frombuffer(result.stdout, dtype=np.float32)
            frames = pcm.size // ch
            if frames > 0:
                return pcm[: frames * ch].reshape(frames, ch), sr

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

            for item in self.selected:
                base = _safe_name(item.name)
                base_key = base.lower()
                if base_key in existing_basenames or base_key in batch_basenames:
                    skipped.append(f"{item.name} (duplicate name)")
                    continue
                suffix = _url_suffix(item.url)
                if suffix not in {".wav", ".mp3"}:
                    suffix = ".wav"
                out_path = self.sounds_dir / f"{base}{suffix}"

                data = _download(item.url)
                if _url_suffix(item.url) in {".wav", ".mp3"}:
                    out_path.write_bytes(data)
                    imported.append(str(out_path))
                    existing_basenames.add(base_key)
                    batch_basenames.add(base_key)
                    continue

                if not ffmpeg_ready:
                    skipped.append(f"{item.name} (non-WAV source and ffmpeg not found)")
                    continue

                src_suffix = Path(urlparse(item.url).path).suffix or ".bin"
                with tempfile.NamedTemporaryFile(delete=False, suffix=src_suffix) as tmp:
                    tmp.write(data)
                    src_path = Path(tmp.name)

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
        src_path: Path | None = None
        temp_src: Path | None = None
        try:
            data = _download(self.item.url)
            src_suffix = Path(urlparse(self.item.url).path).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=src_suffix) as tmp:
                tmp.write(data)
                src_path = Path(tmp.name)
                temp_src = src_path

            audio, sr = _decode_audio_for_preview(src_path)
            if self.volume < 1.0:
                audio = audio * self.volume
            np.clip(audio, -1.0, 1.0, out=audio)
            self.finished.emit({"item": self.item, "audio": audio, "sr": sr})
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if temp_src is not None:
                temp_src.unlink(missing_ok=True)


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


class LocalTileDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option, index) -> None:  # type: ignore[override]
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = option.rect.adjusted(8, 8, -8, -8)
        colors = index.data(int(Qt.ItemDataRole.UserRole))
        if not (isinstance(colors, tuple) and len(colors) == 2):
            colors = ("#4f46e5", "#3730a3")
        c1 = QColor(colors[0])
        c2 = QColor(colors[1])

        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if hovered:
            c1 = c1.lighter(132)
            c2 = c2.lighter(132)

        shadow_rect = rect.adjusted(1, 2, 1, 3)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(7, 14, 31, 105 if hovered else 82))
        painter.drawRoundedRect(shadow_rect, 19, 19)

        grad = QLinearGradient(float(rect.left()), float(rect.top()), float(rect.right()), float(rect.bottom()))
        grad.setColorAt(0.0, c1)
        grad.setColorAt(1.0, c2)
        painter.setBrush(QBrush(grad))

        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        border_color = QColor(255, 255, 255, 238) if selected else QColor(196, 214, 255, 98)
        border_width = 2 if (selected or hovered) else 1
        pen = QPen(border_color, border_width)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 18, 18)

        text = str(index.data(int(Qt.ItemDataRole.DisplayRole)) or "")
        text_rect = rect.adjusted(10, 0, -10, 0)
        text = option.fontMetrics.elidedText(text, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.setPen(QColor("#f8fbff"))
        painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignCenter), text)

        if index.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            state = index.data(int(Qt.ItemDataRole.CheckStateRole))
            checked = state == Qt.CheckState.Checked
            check_rect = QRect(rect.right() - 20, rect.top() + 6, 12, 12)
            painter.setBrush(QColor(255, 255, 255, 230) if checked else QColor(255, 255, 255, 125))
            painter.setPen(QPen(QColor(255, 255, 255, 210), 1))
            painter.drawEllipse(check_rect)
            if checked:
                painter.setPen(QPen(QColor(30, 41, 59), 2))
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
        bg_grad = QLinearGradient(outer_f.left(), outer_f.top(), outer_f.right(), outer_f.bottom())
        bg_grad.setColorAt(0.0, QColor(17, 35, 80, 200))
        bg_grad.setColorAt(1.0, QColor(8, 16, 44, 235))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor(173, 210, 255, 110), 1))
        painter.drawRoundedRect(outer_f, 14, 14)

        track = self._track_rect()
        track_grad = QLinearGradient(track.left(), track.top(), track.left(), track.bottom())
        track_grad.setColorAt(0.0, QColor(36, 72, 140, 125))
        track_grad.setColorAt(1.0, QColor(16, 33, 82, 160))
        painter.setBrush(QBrush(track_grad))
        painter.setPen(QPen(QColor(153, 184, 255, 95), 1))
        painter.drawRoundedRect(track, 10, 10)

        bar_count = max(32, int(track.width() / 7))
        center_y = track.center().y()
        for i in range(bar_count):
            t = i / max(1, bar_count - 1)
            x = track.left() + (t * track.width())
            wave = 0.15 + (abs(math.sin((t * 9.6) + 0.6)) * 0.85)
            amp = (track.height() * wave) * 0.5
            alpha = 55 + int(80 * wave)
            painter.setPen(QPen(QColor(120, 170, 255, alpha), 2))
            painter.drawLine(QPointF(x, center_y - amp), QPointF(x, center_y + amp))

        start_x = self._ms_to_x(self._start_ms)
        end_x = self._ms_to_x(self._end_ms)
        play_x = self._ms_to_x(self._playhead_ms)

        selection = QRectF(start_x, track.top(), max(2.0, end_x - start_x), track.height())
        sel_grad = QLinearGradient(selection.left(), selection.top(), selection.right(), selection.bottom())
        sel_grad.setColorAt(0.0, QColor(42, 212, 255, 95))
        sel_grad.setColorAt(1.0, QColor(64, 120, 255, 105))
        painter.setBrush(QBrush(sel_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(selection, 8, 8)

        def draw_pin(x_pos: float, color: QColor) -> None:
            pin_top = track.top() - 11.0
            pin_bottom = track.bottom() + 8.0
            painter.setPen(QPen(color, 2.2))
            painter.drawLine(QPointF(x_pos, pin_top + 4.0), QPointF(x_pos, pin_bottom))
            painter.setBrush(QBrush(color.lighter(120)))
            painter.setPen(QPen(QColor(244, 251, 255, 220), 1))
            painter.drawEllipse(QPointF(x_pos, pin_top), 5.5, 5.5)

        draw_pin(start_x, QColor(52, 211, 153))
        draw_pin(end_x, QColor(248, 113, 113))
        painter.setPen(QPen(QColor(236, 245, 255, 230), 1.8))
        painter.drawLine(QPointF(play_x, track.top() - 7.0), QPointF(play_x, track.bottom() + 7.0))
        painter.setBrush(QBrush(QColor(236, 245, 255, 235)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(play_x, track.top() - 7.0), 4.0, 4.0)

        painter.setPen(QColor(221, 233, 255, 220))
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
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(13, 27, 66, 245),
                    stop:1 rgba(8, 17, 44, 250));
                border: 1px solid rgba(160, 193, 255, 0.45);
                border-radius: 16px;
            }
            QDialog#TrimEditorDialog QLabel#TrimTarget {
                font-size: 14px;
                font-weight: 700;
                color: #eff6ff;
            }
            QDialog#TrimEditorDialog QLabel#TrimTime {
                font-size: 13px;
                font-weight: 700;
                color: #d5e4ff;
                padding-right: 2px;
            }
            """
        )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.closed.emit()
        super().closeEvent(event)

class SoundboardWindow(QMainWindow):
    DEFAULT_KEYS = list("1234567890qwertyuiopasdfghjklzxcvbnm")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SoundboardEZ")
        base_h = 760
        self.resize(int(base_h * PHI), base_h)

        self.sounds_dir = Path("sounds")
        self._local_all_items: list[str] = []
        self._feed_url = MYINSTANTS_INDEX_URL
        self._feed_page = 0
        self._feed_loading = False
        self._feed_end_reached = False
        self._remote_seen_urls: set[str] = set()
        self._feed_play_buttons: dict[str, QPushButton] = {}
        self._current_preview_url: str | None = None
        self._preview_request_id = 0
        self._threads: list[QThread] = []
        self._workers: list[QObject] = []
        self._fetch_in_progress = False
        self._delete_mode = False
        self._trim_mode = False
        self._trim_source_path: Path | None = None
        self._trim_audio: np.ndarray | None = None
        self._trim_sr = 48000
        self._trim_playing = False
        self._trim_playhead = 0
        self._trim_stream: sd.OutputStream | None = None
        self._trim_preview_gain = 0.05
        self._trim_updating_slider = False
        self._fetch_timeout = QTimer(self)
        self._fetch_timeout.setSingleShot(True)
        self._fetch_timeout.timeout.connect(self._handle_fetch_timeout)
        self._preview_monitor = QTimer(self)
        self._preview_monitor.setInterval(150)
        self._preview_monitor.timeout.connect(self._check_preview_finished)
        self._trim_play_timer = QTimer(self)
        self._trim_play_timer.setInterval(80)
        self._trim_play_timer.timeout.connect(self._sync_trim_playhead)
        self._ui_animations: list[QPropertyAnimation] = []

        self.engine = AudioEngine(
            samplerate=48000,
            blocksize=512,
            input_channels=1,
            output_channels=1,
            sounds_dir=str(self.sounds_dir),
        )
        self._engine_thread = threading.Thread(target=self._run_engine, daemon=True)
        self._engine_thread.start()

        self._button_motion = ButtonMotionFilter(self)

        central = QWidget()
        central.setObjectName("Root")
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        grid.setContentsMargins(18, 14, 18, 14)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)
        grid.setRowStretch(1, 1)

        self.top_bar = QFrame()
        self.top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(16, 10, 16, 10)
        top_layout.setSpacing(10)
        self.app_logo = QLabel("SB")
        self.app_logo.setObjectName("LogoBadge")
        self.app_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(0)
        self.app_title = QLabel("SoundboardEZ")
        self.app_title.setObjectName("AppTitle")
        self.app_subtitle = QLabel("Virtual Mic Mixer")
        self.app_subtitle.setObjectName("AppSubtitle")
        title_col.addWidget(self.app_title)
        title_col.addWidget(self.app_subtitle)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusPill")
        top_layout.addWidget(self.app_logo)
        top_layout.addLayout(title_col)
        top_layout.addStretch(1)
        top_layout.addWidget(self.status_label, 0, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.top_bar, 0, 0)

        self.left_group = QGroupBox("Importer")
        self.left_group.setObjectName("ImporterCard")
        self.left_group.setMinimumWidth(430)
        self.left_group.setMaximumWidth(500)
        left_layout = QGridLayout(self.left_group)
        left_layout.setContentsMargins(14, 18, 14, 12)
        left_layout.setHorizontalSpacing(8)
        left_layout.setVerticalSpacing(8)
        self.import_search_input = QLineEdit()
        self.import_search_input.setPlaceholderText("Search tags or paste Google/myinstants link")
        self.import_search_btn = QPushButton("Search")
        self.import_search_btn.setProperty("variant", "primary")
        self.import_url_btn = QPushButton("Import Link")
        self.import_url_btn.setProperty("variant", "success")
        self.close_importer_btn = QPushButton("X")
        self.close_importer_btn.setObjectName("ImporterClose")
        self.close_importer_btn.setToolTip("Close importer")
        self.fetch_btn = QPushButton("Reload Feed")
        self.fetch_btn.setProperty("variant", "secondary")
        self.import_file_btn = QPushButton("Import File")
        self.import_file_btn.setProperty("variant", "primary")
        self.preview_volume_label = QLabel("Preview Volume: 8%")
        self.preview_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_volume_slider.setRange(0, 100)
        self.preview_volume_slider.setValue(8)
        self.remote_feed_list = SmoothListWidget(slow_factor=0.5)
        self.remote_feed_list.setSpacing(14)

        left_layout.addWidget(QLabel("Import Search"), 0, 0)
        left_layout.addWidget(self.import_search_input, 0, 1, 1, 2)
        left_layout.addWidget(self.import_search_btn, 0, 3)
        left_layout.addWidget(self.import_url_btn, 0, 4)
        left_layout.addWidget(self.close_importer_btn, 0, 5)
        left_layout.addWidget(self.import_file_btn, 1, 4)
        left_layout.addWidget(self.fetch_btn, 1, 3)
        left_layout.addWidget(self.preview_volume_label, 2, 0, 1, 2)
        left_layout.addWidget(self.preview_volume_slider, 2, 2, 1, 4)
        importer_hint = QLabel("Click Play or Import on any sound. More loads as you scroll.")
        importer_hint.setObjectName("HintLabel")
        left_layout.addWidget(importer_hint, 3, 0, 1, 6)
        left_layout.addWidget(self.remote_feed_list, 4, 0, 1, 6)

        self.right_group = QGroupBox("Your Soundboard")
        self.right_group.setObjectName("MainCard")
        right_layout = QHBoxLayout(self.right_group)
        right_layout.setContentsMargins(14, 18, 14, 12)
        right_layout.setSpacing(12)
        self.soundboard_sidebar = QWidget()
        self.soundboard_sidebar.setObjectName("SideCard")
        sidebar_w = 268
        self.soundboard_sidebar.setMinimumWidth(sidebar_w)
        self.soundboard_sidebar.setMaximumWidth(sidebar_w)
        sidebar_layout = QVBoxLayout(self.soundboard_sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        self.soundboard_main = QWidget()
        self.soundboard_main.setObjectName("SoundboardMain")
        main_layout = QVBoxLayout(self.soundboard_main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        self.soundboard_volume_label = QLabel("Soundboard Volume: 100%")
        self.soundboard_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.soundboard_volume_slider.setRange(0, 200)
        self.soundboard_volume_slider.setValue(70)
        self.speaker_monitor_btn = QPushButton("Play To Speaker: Off")
        self.speaker_monitor_btn.setProperty("variant", "primary")
        self.speaker_monitor_btn.setCheckable(True)
        self.speaker_monitor_label = QLabel("Speaker Volume: 2%")
        self.speaker_monitor_slider = QSlider(Qt.Orientation.Horizontal)
        self.speaker_monitor_slider.setRange(0, 100)
        self.speaker_monitor_slider.setValue(2)
        self.speaker_monitor_label.setVisible(False)
        self.speaker_monitor_slider.setVisible(False)
        self.toggle_importer_btn = QPushButton("Open Importer")
        self.toggle_importer_btn.setProperty("variant", "violet")
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
        self.local_search_input.setPlaceholderText("Search imported sounds")
        self.local_list = SmoothListWidget(slow_factor=0.6)
        self.local_list.setObjectName("LocalSoundGrid")
        self.local_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.local_list.setViewMode(QListView.ViewMode.IconMode)
        self.local_list.setFlow(QListView.Flow.LeftToRight)
        self.local_list.setWrapping(True)
        self.local_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.local_list.setMovement(QListView.Movement.Static)
        self.local_list.setUniformItemSizes(True)
        self.local_list.setSpacing(20)
        self.local_list.setWordWrap(True)
        self.local_list.setSelectionRectVisible(False)
        self.local_list.setMouseTracking(True)
        self.local_list.setItemDelegate(LocalTileDelegate(self.local_list))
        self.local_title = QLabel("Imported Sounds")
        self.local_title.setObjectName("SectionTitle")
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
        sidebar_layout.addWidget(self.toggle_importer_btn)
        sidebar_layout.addWidget(self.soundboard_volume_label)
        sidebar_layout.addWidget(self.soundboard_volume_slider)
        sidebar_layout.addWidget(self.speaker_monitor_btn)
        sidebar_layout.addWidget(self.speaker_monitor_label)
        sidebar_layout.addWidget(self.speaker_monitor_slider)
        sidebar_layout.addWidget(self.play_local_btn)
        sidebar_layout.addWidget(self.trim_local_btn)
        sidebar_layout.addWidget(self.cancel_trim_btn)
        sidebar_layout.addWidget(self.delete_local_btn)
        sidebar_layout.addWidget(self.cancel_delete_btn)
        sidebar_layout.addWidget(self.refresh_local_btn)
        sidebar_layout.addWidget(self.delete_mode_hint)
        sidebar_layout.addWidget(self.trim_mode_hint)
        sidebar_layout.addStretch(1)

        local_header = QHBoxLayout()
        local_header.setSpacing(10)
        local_header.addWidget(self.local_title)
        local_header.addStretch(1)
        local_header.addWidget(self.local_search_input, 0)
        main_layout.addLayout(local_header)
        main_layout.addWidget(self.local_list, 1)

        right_layout.addWidget(self.soundboard_sidebar)
        right_layout.addWidget(self.soundboard_main, 1)

        top_row = QHBoxLayout()
        top_row.setSpacing(14)
        top_row.addWidget(self.right_group, 1618)
        top_row.addWidget(self.left_group, 1000)
        grid.addLayout(top_row, 1, 0)
        self._importer_fx = QGraphicsOpacityEffect(self.left_group)
        self.left_group.setGraphicsEffect(self._importer_fx)
        self._importer_fx.setOpacity(1.0)
        self._importer_anim = QPropertyAnimation(self._importer_fx, b"opacity", self)
        self._importer_anim.setDuration(190)
        self._importer_anim.setStartValue(0.0)
        self._importer_anim.setEndValue(1.0)
        self._importer_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.import_search_btn.clicked.connect(self.apply_import_search)
        self.import_url_btn.clicked.connect(self.import_from_link)
        self.close_importer_btn.clicked.connect(self.close_importer_panel)
        self.import_search_input.returnPressed.connect(self.apply_import_search)
        self.fetch_btn.clicked.connect(self.fetch_sounds)
        self.preview_volume_slider.valueChanged.connect(self.update_preview_volume_label)
        self.soundboard_volume_slider.valueChanged.connect(self.update_soundboard_volume)
        self.speaker_monitor_btn.toggled.connect(self.toggle_speaker_monitor)
        self.speaker_monitor_slider.valueChanged.connect(self.update_speaker_monitor_volume_label)
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
        self._importer_loaded_once = False
        self._set_importer_visible(False)
        self.refresh_local()
        self.update_preview_volume_label(self.preview_volume_slider.value())
        self.update_soundboard_volume(self.soundboard_volume_slider.value())
        self.update_speaker_monitor_volume_label(self.speaker_monitor_slider.value())
        self._run_entrance_animation()

    def _run_engine(self) -> None:
        try:
            mapping = self.engine.setup_soundboard(auto_hotkeys=True)
            if mapping:
                print("Initial hotkeys:")
                for key, name in mapping.items():
                    print(f"  {key} -> {name}")
            self.engine.start()
        except Exception as exc:
            print(f"Audio engine error: {exc}")

    def _apply_button_ratios(self) -> None:
        sidebar_buttons = [
            self.toggle_importer_btn,
            self.speaker_monitor_btn,
            self.play_local_btn,
            self.trim_local_btn,
            self.cancel_trim_btn,
            self.delete_local_btn,
            self.cancel_delete_btn,
            self.refresh_local_btn,
        ]
        for btn in sidebar_buttons:
            btn.setMinimumHeight(38)
            btn.setMaximumHeight(40)
            btn.setMinimumWidth(220)

        importer_buttons = [
            self.import_search_btn,
            self.import_url_btn,
            self.fetch_btn,
            self.import_file_btn,
        ]
        for btn in importer_buttons:
            btn.setMinimumHeight(34)
            btn.setMaximumHeight(36)
            btn.setMinimumWidth(96)
            btn.setMaximumWidth(130)

        self.close_importer_btn.setMinimumSize(30, 30)
        self.close_importer_btn.setMaximumSize(30, 30)
        self.local_search_input.setMinimumWidth(260)
        self.import_search_input.setMinimumHeight(36)

    def _apply_modern_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget#Root {
                background: qradialgradient(cx:0.16, cy:0.08, radius:1.28,
                    fx:0.08, fy:0.05,
                    stop:0 rgba(48, 89, 255, 0.42),
                    stop:0.34 rgba(13, 23, 56, 0.98),
                    stop:1 rgba(4, 8, 24, 1.0));
                color: #edf4ff;
                font-family: "SF Pro Display", "Segoe UI Variable Text", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QFrame#TopBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 255, 255, 0.12),
                    stop:1 rgba(255, 255, 255, 0.05));
                border: 1px solid rgba(180, 207, 255, 0.22);
                border-radius: 18px;
            }
            QLabel#LogoBadge {
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
                border-radius: 16px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(94, 234, 212, 0.95),
                    stop:1 rgba(99, 102, 241, 0.95));
                color: #061123;
                font-size: 12px;
                font-weight: 800;
            }
            QLabel#AppTitle {
                color: #f8fbff;
                font-size: 17px;
                font-weight: 750;
                letter-spacing: 0.3px;
            }
            QLabel#AppSubtitle {
                color: rgba(220, 233, 255, 0.78);
                font-size: 12px;
                font-weight: 500;
            }
            QGroupBox#MainCard,
            QGroupBox#ImporterCard,
            QGroupBox#SideCard {
                background: qlineargradient(x1:0, y1:0, x2:0.9, y2:1,
                    stop:0 rgba(255, 255, 255, 0.10),
                    stop:1 rgba(255, 255, 255, 0.04));
                border: 1px solid rgba(190, 214, 255, 0.24);
                border-radius: 20px;
                margin-top: 10px;
                padding-top: 11px;
            }
            QGroupBox#MainCard::title,
            QGroupBox#ImporterCard::title,
            QGroupBox#SideCard::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 3px 12px;
                color: #f3f8ff;
                font-weight: 700;
                background: rgba(10, 22, 54, 0.92);
                border: 1px solid rgba(124, 223, 255, 0.42);
                border-radius: 11px;
            }
            QWidget#SideCard {
                background: rgba(9, 21, 56, 0.50);
                border-radius: 18px;
            }
            QWidget#SoundboardMain {
                background: rgba(7, 16, 43, 0.48);
                border: 1px solid rgba(167, 196, 255, 0.20);
                border-radius: 18px;
                padding: 6px;
            }
            QLabel#SectionTitle {
                font-size: 18px;
                font-weight: 730;
                color: #f6fbff;
                letter-spacing: 0.2px;
            }
            QLabel {
                color: #dbe7ff;
            }
            QLabel#HintLabel {
                color: rgba(212, 229, 255, 0.78);
                font-size: 11px;
            }
            QLabel#FeedNameLabel {
                color: #edf4ff;
                font-size: 13px;
                font-weight: 560;
            }
            QLabel#StatusPill {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(12, 31, 80, 0.86),
                    stop:1 rgba(20, 58, 131, 0.72));
                color: #f3f8ff;
                border: 1px solid rgba(128, 229, 255, 0.65);
                border-radius: 14px;
                padding: 8px 13px;
                font-weight: 650;
            }
            QLineEdit {
                background: rgba(9, 20, 49, 0.62);
                border: 1px solid rgba(143, 187, 255, 0.35);
                border-radius: 14px;
                padding: 8px 12px;
                selection-background-color: rgba(83, 204, 255, 0.65);
                color: #eef4ff;
            }
            QLineEdit:focus {
                border: 1px solid rgba(91, 219, 255, 0.92);
                background: rgba(13, 26, 68, 0.88);
            }
            QPushButton {
                border-radius: 18px;
                border: 1px solid rgba(194, 220, 255, 0.34);
                padding: 6px 14px;
                min-height: 34px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255,255,255,0.23),
                    stop:1 rgba(255,255,255,0.10));
                color: #f5f8ff;
                font-weight: 680;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255,255,255,0.30),
                    stop:1 rgba(255,255,255,0.16));
                border-color: rgba(132, 238, 255, 0.74);
            }
            QPushButton:pressed {
                padding-top: 8px;
                padding-bottom: 4px;
                background: rgba(255, 255, 255, 0.23);
            }
            QPushButton[variant="primary"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(86, 197, 255, 0.68),
                    stop:1 rgba(36, 112, 255, 0.58));
                border-color: rgba(143, 225, 255, 0.90);
            }
            QPushButton[variant="success"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(34, 219, 169, 0.68),
                    stop:1 rgba(11, 163, 124, 0.56));
                border-color: rgba(119, 255, 216, 0.86);
            }
            QPushButton[variant="danger"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 106, 146, 0.70),
                    stop:1 rgba(215, 48, 96, 0.58));
                border-color: rgba(255, 182, 204, 0.90);
            }
            QPushButton[variant="violet"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(178, 121, 255, 0.72),
                    stop:1 rgba(118, 74, 224, 0.58));
                border-color: rgba(216, 188, 255, 0.90);
            }
            QPushButton[variant="cyan"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(55, 219, 255, 0.72),
                    stop:1 rgba(10, 149, 202, 0.58));
                border-color: rgba(150, 242, 255, 0.90);
            }
            QPushButton[variant="amber"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(253, 211, 75, 0.72),
                    stop:1 rgba(217, 126, 20, 0.56));
                border-color: rgba(254, 226, 183, 0.90);
            }
            QPushButton[variant="slate"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(129, 145, 171, 0.58),
                    stop:1 rgba(86, 99, 128, 0.52));
                border-color: rgba(213, 223, 238, 0.76);
            }
            QPushButton#ImporterClose {
                min-width: 30px;
                max-width: 30px;
                min-height: 30px;
                max-height: 30px;
                border-radius: 15px;
                padding: 0;
                background: rgba(255, 94, 126, 0.60);
                border: 1px solid rgba(255, 170, 192, 0.85);
                color: #fff1f5;
                font-weight: 700;
            }
            QPushButton#ImporterClose:hover {
                background: rgba(255, 86, 117, 0.82);
            }
            QListWidget {
                background: rgba(6, 14, 39, 0.60);
                border: 1px solid rgba(160, 195, 255, 0.26);
                border-radius: 18px;
                padding: 8px;
                outline: none;
            }
            QListWidget#LocalSoundGrid {
                background: rgba(5, 12, 34, 0.56);
                border: 1px solid rgba(152, 188, 255, 0.26);
                border-radius: 20px;
                padding: 12px;
            }
            QListWidget#LocalSoundGrid::item {
                border: 1px solid rgba(196, 214, 255, 0.30);
                border-radius: 16px;
                padding: 8px 10px;
                margin: 4px;
                color: #f8fbff;
                font-weight: 700;
            }
            QListWidget#LocalSoundGrid::item:selected {
                border: 1px solid rgba(255, 255, 255, 0.86);
                background: transparent;
                color: #ffffff;
            }
            QListWidget::item {
                border-radius: 10px;
                padding: 6px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background: rgba(66, 138, 255, 0.34);
                color: #ffffff;
            }
            QSlider::groove:horizontal {
                border-radius: 7px;
                height: 10px;
                background: rgba(120, 150, 218, 0.30);
            }
            QSlider::sub-page:horizontal {
                border-radius: 7px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(78, 233, 255, 0.96),
                    stop:1 rgba(96, 128, 255, 0.96));
            }
            QSlider::handle:horizontal {
                background: #f8fbff;
                border: 1px solid rgba(158, 198, 255, 0.95);
                width: 19px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 3px;
            }
            QScrollBar::handle:vertical {
                background: rgba(132, 171, 247, 0.62);
                border: 1px solid rgba(185, 218, 255, 0.62);
                border-radius: 5px;
                min-height: 34px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

    def _apply_button_motion(self) -> None:
        buttons = [
            self.toggle_importer_btn,
            self.speaker_monitor_btn,
            self.play_local_btn,
            self.trim_local_btn,
            self.cancel_trim_btn,
            self.delete_local_btn,
            self.cancel_delete_btn,
            self.refresh_local_btn,
            self.import_search_btn,
            self.import_url_btn,
            self.fetch_btn,
            self.import_file_btn,
            self.close_importer_btn,
            self.trim_play_pause_btn,
            self.trim_stop_btn,
            self.apply_trim_btn,
            self.close_trim_editor_btn,
        ]
        for btn in buttons:
            variant = str(btn.property("variant") or "")
            if variant == "danger":
                glow = "#fb7185"
            elif variant == "success":
                glow = "#34d399"
            elif variant == "violet":
                glow = "#a78bfa"
            elif variant == "cyan":
                glow = "#22d3ee"
            else:
                glow = "#67e8f9"
            self._button_motion.attach(btn, glow=glow)

    def _run_entrance_animation(self) -> None:
        sequence = [
            (self.top_bar, 0),
            (self.soundboard_sidebar, 70),
            (self.right_group, 120),
            (self.left_group, 180),
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
            self._ui_animations.append(anim)
            QTimer.singleShot(start_ms, anim.start)

    def _run_worker(self, worker: QObject, on_finished, on_error) -> None:
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
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

    def _set_importer_visible(self, visible: bool) -> None:
        if visible:
            self.left_group.setVisible(True)
            self._importer_anim.stop()
            self._importer_fx.setOpacity(0.0)
            self._importer_anim.start()
        else:
            self.left_group.setVisible(False)
        self.toggle_importer_btn.setText("Close Importer" if visible else "Open Importer")
        if visible and not self._importer_loaded_once:
            self.fetch_sounds()
            self._importer_loaded_once = True

    def toggle_importer_panel(self) -> None:
        self._set_importer_visible(not self.left_group.isVisible())

    def close_importer_panel(self) -> None:
        self._set_importer_visible(False)

    def apply_import_search(self) -> None:
        raw = self.import_search_input.text().strip()
        resolved, error = _resolve_myinstants_feed_url(raw)
        if error or resolved is None:
            self.status_label.setText(error or "Invalid search input.")
            return
        self.import_search_input.setText(resolved)
        self.fetch_sounds()

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
            self._run_worker(worker, self._on_import_done, self._on_import_error)
            return

        if "myinstants.com" in host:
            audio_url = raw
            if "/media/sounds/" not in parsed.path:
                self.status_label.setText("For myinstants link import, use direct media link or feed Import button.")
                return
            name = Path(parsed.path).stem or "myinstants_sound"
            worker = ImportWorker([RemoteSoundItem(name=name, url=audio_url)], self.sounds_dir)
            self._run_worker(worker, self._on_import_done, self._on_import_error)
            return

        self.status_label.setText("Only YouTube and myinstants links are supported.")

    def import_from_file_dialog(self) -> None:
        self.sounds_dir.mkdir(parents=True, exist_ok=True)
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Audio Files",
            "",
            "Audio Files (*.mp3 *.wav *.ogg *.m4a *.aac *.flac);;All Files (*.*)",
        )
        if not files:
            return

        imported: list[str] = []
        skipped: list[str] = []
        existing_basenames = {p.stem.lower() for p in self.sounds_dir.iterdir() if p.is_file()}
        for src_str in files:
            src = Path(src_str)
            if not src.exists():
                skipped.append(f"{src.name} (missing)")
                continue
            base = _safe_name(src.stem)
            if base.lower() in existing_basenames:
                skipped.append(f"{src.name} (duplicate name)")
                continue
            dst = self.sounds_dir / f"{base}{src.suffix.lower()}"
            try:
                shutil.copy2(src, dst)
                imported.append(str(dst))
                existing_basenames.add(base.lower())
            except Exception as exc:
                skipped.append(f"{src.name} ({exc})")

        self._finalize_import(imported, skipped)

    def fetch_sounds(self) -> None:
        raw = self.import_search_input.text().strip() or MYINSTANTS_INDEX_URL
        url, error = _resolve_myinstants_feed_url(raw)
        if error or url is None:
            self.status_label.setText(error or "Invalid URL.")
            return
        self.import_search_input.setText(url)

        self._feed_url = url
        self._feed_page = 0
        self._feed_end_reached = False
        self._remote_seen_urls.clear()
        self._feed_play_buttons.clear()
        self.remote_feed_list.clear()
        self._load_next_feed_page()

    def _load_next_feed_page(self) -> None:
        if self._feed_loading or self._feed_end_reached:
            return

        next_page = self._feed_page + 1
        self._feed_loading = True
        self._fetch_in_progress = True
        self._fetch_timeout.start(20000)
        self.fetch_btn.setEnabled(False)
        self.status_label.setText(f"Loading sounds page {next_page}...")
        worker = FetchWorker(self._feed_url, next_page)

        def done(rows: list[tuple[str, str]]) -> None:
            self._fetch_in_progress = False
            self._feed_loading = False
            self._fetch_timeout.stop()
            self.fetch_btn.setEnabled(True)
            if not rows:
                if next_page == 1:
                    self.status_label.setText("No sounds found on this page URL.")
                else:
                    self.status_label.setText("Reached end of available sounds.")
                self._feed_end_reached = True
                return

            added = 0
            for name, url in rows:
                if url in self._remote_seen_urls:
                    continue
                self._remote_seen_urls.add(url)
                self._add_feed_row(RemoteSoundItem(name=name, url=url))
                added += 1

            self._feed_page = next_page
            if added == 0:
                self._feed_end_reached = True
                self.status_label.setText("No new sounds on next page.")
            else:
                self.status_label.setText(f"Loaded page {next_page} ({added} sounds). Scroll for more.")

        def err(message: str) -> None:
            self._fetch_in_progress = False
            self._feed_loading = False
            self._fetch_timeout.stop()
            self.fetch_btn.setEnabled(True)
            self.status_label.setText("Feed load failed.")
            QMessageBox.critical(self, "Fetch Error", message)

        self._run_worker(worker, done, err)

    def _on_feed_scroll(self, value: int) -> None:
        bar = self.remote_feed_list.verticalScrollBar()
        if bar.maximum() <= 0:
            return
        if value >= bar.maximum() - 80:
            self._load_next_feed_page()

    def _add_feed_row(self, item: RemoteSoundItem) -> None:
        row_item = QListWidgetItem()
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(10, 6, 10, 6)
        row_layout.setSpacing(10)
        row_widget.setMinimumHeight(46)

        name_lbl = QLabel(item.name)
        name_lbl.setObjectName("FeedNameLabel")
        name_lbl.setToolTip(item.url)
        play_btn = QPushButton("Play")
        row_btn_h = 30
        play_btn_w = 84
        import_btn_w = 112
        play_btn.setFixedSize(play_btn_w, row_btn_h)
        import_btn = QPushButton("Import")
        import_btn.setFixedSize(import_btn_w, row_btn_h)
        self._style_feed_buttons(play_btn, import_btn, seed=item.url)
        play_btn.clicked.connect(lambda _=False, it=item: self.toggle_remote_play(it))
        import_btn.clicked.connect(lambda _=False, it=item: self.import_remote_item(it))
        self._feed_play_buttons[item.url] = play_btn

        row_layout.addWidget(name_lbl, 1)
        row_layout.addWidget(play_btn)
        row_layout.addWidget(import_btn)

        row_item.setSizeHint(QSize(0, 58))
        row_widget.mouseDoubleClickEvent = lambda _event, it=item: self.toggle_remote_play(it)  # type: ignore[attr-defined]
        self.remote_feed_list.addItem(row_item)
        self.remote_feed_list.setItemWidget(row_item, row_widget)

    def _style_feed_buttons(self, play_btn: QPushButton, import_btn: QPushButton, seed: str) -> None:
        play_palette = [
            ("#60a5fa", "#2563eb"),
            ("#22d3ee", "#0891b2"),
            ("#a78bfa", "#7c3aed"),
            ("#f472b6", "#db2777"),
            ("#f59e0b", "#d97706"),
        ]
        import_palette = [
            ("#34d399", "#059669"),
            ("#22c55e", "#15803d"),
            ("#2dd4bf", "#0f766e"),
            ("#f97316", "#ea580c"),
            ("#fb7185", "#e11d48"),
        ]
        rng = random.Random(seed)
        pc1, pc2 = play_palette[rng.randrange(len(play_palette))]
        ic1, ic2 = import_palette[rng.randrange(len(import_palette))]

        play_btn.setStyleSheet(
            f"""
            QPushButton {{
                border-radius: 15px;
                border: 1px solid rgba(220,235,255,0.80);
                font-size: 12px;
                font-weight: 700;
                min-height: 0px;
                max-height: 30px;
                padding: 0px 10px;
                color: #f8fbff;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {pc1}, stop:1 {pc2});
            }}
            QPushButton:hover {{
                border-color: rgba(255,255,255,0.98);
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255,255,255,0.36), stop:1 {pc2});
            }}
            QPushButton:pressed {{ padding-top: 2px; background: rgba(255,255,255,0.22); }}
            QPushButton:disabled {{ background: rgba(107,114,128,0.55); color: rgba(255,255,255,0.8); }}
            """
        )
        import_btn.setStyleSheet(
            f"""
            QPushButton {{
                border-radius: 15px;
                border: 1px solid rgba(220,255,240,0.86);
                font-size: 12px;
                font-weight: 700;
                min-height: 0px;
                max-height: 30px;
                padding: 0px 10px;
                color: #f8fbff;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {ic1}, stop:1 {ic2});
            }}
            QPushButton:hover {{
                border-color: rgba(255,255,255,0.98);
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255,255,255,0.34), stop:1 {ic2});
            }}
            QPushButton:pressed {{ padding-top: 2px; background: rgba(255,255,255,0.22); }}
            """
        )
        self._button_motion.attach(play_btn, glow=pc1)
        self._button_motion.attach(import_btn, glow=ic1)

    def toggle_remote_play(self, item: RemoteSoundItem) -> None:
        if self._current_preview_url == item.url:
            self.stop_remote_preview()
            return
        self.start_remote_preview(item)

    def start_remote_preview(self, item: RemoteSoundItem) -> None:
        self.stop_remote_preview(silent=True)
        self._preview_request_id += 1
        request_id = self._preview_request_id
        self._set_feed_play_button(item.url, "Loading...", enabled=False)
        self.status_label.setText(f"Loading '{item.name}'...")
        worker = PreviewWorker(item, volume=self.preview_volume_slider.value() / 100.0)

        def done(payload: object) -> None:
            if request_id != self._preview_request_id:
                return
            data = payload  # type: ignore[assignment]
            try:
                remote_item = data["item"]
                audio = data["audio"]
                sr = data["sr"]
                sd.play(audio, samplerate=sr, blocking=False)
            except Exception as exc:
                self._reset_feed_play_buttons()
                self.status_label.setText("Preview failed.")
                QMessageBox.warning(self, "Preview Error", str(exc))
                return

            self._current_preview_url = remote_item.url
            self._reset_feed_play_buttons()
            self._set_feed_play_button(remote_item.url, "Stop", enabled=True)
            self._preview_monitor.start()
            self.status_label.setText(f"Previewing '{remote_item.name}'.")

        def err(message: str) -> None:
            if request_id != self._preview_request_id:
                return
            self._reset_feed_play_buttons()
            self.status_label.setText("Preview failed.")
            QMessageBox.warning(self, "Preview Error", message)

        self._run_worker(worker, done, err)

    def stop_remote_preview(self, silent: bool = False) -> None:
        self._preview_request_id += 1
        try:
            sd.stop()
        except Exception:
            pass
        self._current_preview_url = None
        self._preview_monitor.stop()
        self._reset_feed_play_buttons()
        if not silent:
            self.status_label.setText("Preview stopped.")

    def _set_feed_play_button(self, url: str, text: str, enabled: bool = True) -> None:
        btn = self._feed_play_buttons.get(url)
        if btn is not None:
            btn.setText(text)
            btn.setEnabled(enabled)

    def _reset_feed_play_buttons(self) -> None:
        for btn in self._feed_play_buttons.values():
            btn.setText("Play")
            btn.setEnabled(True)

    def _check_preview_finished(self) -> None:
        if self._current_preview_url is None:
            self._preview_monitor.stop()
            return
        try:
            stream = sd.get_stream()
            active = bool(stream is not None and stream.active)
        except Exception:
            active = False
        if not active:
            self._current_preview_url = None
            self._preview_monitor.stop()
            self._reset_feed_play_buttons()
            self.status_label.setText("Preview finished.")

    def import_remote_item(self, item: RemoteSoundItem) -> None:
        self.stop_remote_preview(silent=True)
        self.status_label.setText(f"Importing '{item.name}'...")
        worker = ImportWorker([item], self.sounds_dir)
        self._run_worker(worker, self._on_import_done, self._on_import_error)

    def _handle_fetch_timeout(self) -> None:
        if not self._fetch_in_progress:
            return
        self._fetch_in_progress = False
        self._feed_loading = False
        self.fetch_btn.setEnabled(True)
        self.status_label.setText("Fetch timed out. Try reload.")

    def _on_import_done(self, result: dict) -> None:
        imported = result.get("imported", [])
        skipped = result.get("skipped", [])
        self._finalize_import(imported, skipped)

    def _on_import_error(self, message: str) -> None:
        self.status_label.setText("Import failed.")
        QMessageBox.critical(self, "Import Error", message)

    def _finalize_import(self, imported: list[str], skipped: list[str]) -> None:
        self.refresh_local()
        self._register_new_sounds(imported)
        if imported:
            self.status_label.setText(f"Imported {len(imported)} sound(s).")
        elif skipped:
            self.status_label.setText(skipped[0])
        else:
            self.status_label.setText("Nothing imported.")

    def _register_new_sounds(self, imported_paths: list[str]) -> None:
        if not imported_paths:
            return

        existing_hotkeys = set(self.engine.soundboard.hotkeys.keys())
        free_keys = [k for k in self.DEFAULT_KEYS if k not in existing_hotkeys]

        for path_str in imported_paths:
            path = Path(path_str)
            name = path.stem
            try:
                self.engine.soundboard.load_audio_file(name, path)
            except Exception as exc:
                print(f"Failed to load imported sound '{name}': {exc}")
                continue

            if free_keys:
                key = free_keys.pop(0)
                try:
                    self.engine.soundboard.bind_hotkey(key, name)
                except Exception as exc:
                    print(f"Failed to bind hotkey '{key}' to '{name}': {exc}")

    def refresh_local(self) -> None:
        self.sounds_dir.mkdir(parents=True, exist_ok=True)
        self._local_all_items.clear()
        for ext in ("*.wav", "*.mp3"):
            for audio_path in sorted(self.sounds_dir.glob(ext)):
                self._local_all_items.append(audio_path.name)
        self._update_local_grid_size()
        self.apply_local_filter()

    def apply_local_filter(self, _text: str | None = None) -> None:
        query = self.local_search_input.text().strip().lower()
        self.local_list.clear()
        tile_w = max(90, self.local_list.gridSize().width() - 12)
        tile_h = int(tile_w / PHI)
        visual_idx = 0
        for name in self._local_all_items:
            if query and query not in name.lower():
                continue
            item = QListWidgetItem(name)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setSizeHint(QSize(tile_w, tile_h))
            self._style_local_tile(item, visual_idx)
            visual_idx += 1
            # Force non-checkable baseline; delete mode enables this explicitly.
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            self.local_list.addItem(item)
        if self._delete_mode:
            self._set_delete_checkboxes(True)

    def _style_local_tile(self, item: QListWidgetItem, idx: int) -> None:
        key = item.text().strip().lower()
        colors = self._local_tile_colors.get(key)
        if colors is None:
            colors = self._build_random_tile_colors(key)
            self._local_tile_colors[key] = colors
        item.setData(int(Qt.ItemDataRole.UserRole), colors)
        item.setForeground(QBrush(QColor("#f8fbff")))
        item.setToolTip(item.text())

    def _build_random_tile_colors(self, seed_text: str) -> tuple[str, str]:
        rng = random.Random(seed_text)
        h = rng.randint(0, 359)
        s1 = rng.randint(145, 225)
        v1 = rng.randint(185, 250)
        s2 = min(255, s1 + rng.randint(10, 26))
        v2 = max(85, v1 - rng.randint(45, 90))
        c1 = QColor.fromHsv(h, s1, v1)
        c2 = QColor.fromHsv((h + rng.randint(14, 34)) % 360, s2, v2)
        return (c1.name(), c2.name())

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

    def _update_local_grid_size(self) -> None:
        cols = 5
        spacing = self.local_list.spacing()
        view_w = max(480, self.local_list.viewport().width())
        tile_w = int((view_w - (spacing * (cols + 1))) / cols)
        tile_w = max(int(62 * PHI), tile_w)
        tile_h = max(44, int(tile_w / (PHI * 1.32)))
        self.local_list.setGridSize(QSize(tile_w, tile_h + 12))

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
        file_name = item.text().strip()
        file_path = self.sounds_dir / file_name
        if not file_path.exists():
            QMessageBox.warning(self, "Play Imported", f"File not found: {file_name}")
            self.refresh_local()
            return

        sound_name = file_path.stem
        try:
            if sound_name not in self.engine.soundboard.sounds:
                self.engine.soundboard.load_audio_file(sound_name, file_path)
            self.engine.soundboard.trigger(sound_name)
            self._play_sound_to_speaker_if_enabled(sound_name)
            self.status_label.setText(f"Playing '{sound_name}' to virtual mic.")
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
        self._animate_local_tile_click(item)
        if self._trim_mode:
            self._open_trim_editor_for_item(item)

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

    def _open_trim_editor_for_item(self, item: QListWidgetItem) -> None:
        file_name = item.text().strip()
        src = self.sounds_dir / file_name
        if not src.exists():
            self.status_label.setText("Selected file does not exist.")
            self.refresh_local()
            return
        try:
            audio, sr = _decode_audio_for_preview(src)
        except Exception as exc:
            self.status_label.setText(f"Failed to load trim audio: {exc}")
            return

        self.stop_trim_preview(silent=True)
        self._trim_source_path = src
        self._trim_audio = audio
        self._trim_sr = sr
        self._trim_playhead = 0
        total_frames = max(1, audio.shape[0])
        max_ms = max(1, int((total_frames / sr) * 1000))
        self._trim_updating_slider = True
        self.trim_dialog.timeline.set_duration_ms(max_ms)
        self.trim_dialog.timeline.set_range_ms(0, max_ms, emit=False)
        self.trim_dialog.timeline.set_playhead_ms(0, emit=False)
        self._trim_updating_slider = False
        self.trim_dialog.trim_target_label.setText(file_name)
        self.trim_dialog.trim_time_label.setText(f"00:00.00 / {self._format_ms(max_ms)}")
        self.trim_play_pause_btn.setText("Play")
        if not self.trim_dialog.isVisible():
            dialog_geo = self.trim_dialog.frameGeometry()
            dialog_geo.moveCenter(self.frameGeometry().center())
            self.trim_dialog.move(dialog_geo.topLeft())
            self.trim_dialog.show()
        self.trim_dialog.raise_()
        self.trim_dialog.activateWindow()
        self.status_label.setText(f"Trim editor opened for '{file_name}'.")

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
        with tempfile.NamedTemporaryFile(delete=False, suffix=src.suffix or ".wav") as tmp:
            trimmed_tmp = Path(tmp.name)
        cmd = [
            ffmpeg_exe,
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{length:.3f}",
            "-i",
            str(src),
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
            self.status_label.setText("Failed to trim selected audio.")
            return

        self.stop_trim_preview(silent=True, reset_to_start=False)
        try:
            trimmed_tmp.replace(src)
        except Exception as exc:
            trimmed_tmp.unlink(missing_ok=True)
            self.status_label.setText(f"Failed to overwrite '{src.name}': {exc}")
            return

        try:
            self.engine.soundboard.load_audio_file(src.stem, src)
        except Exception as exc:
            self.status_label.setText(f"Trim saved, but failed to reload sound: {exc}")
            return

        try:
            audio, sr = _decode_audio_for_preview(src)
        except Exception as exc:
            self.status_label.setText(f"Trim saved, but failed to reopen editor: {exc}")
            return

        self._trim_source_path = src
        self._trim_audio = audio
        self._trim_sr = sr
        self._trim_playhead = 0
        max_ms = max(1, int((max(1, audio.shape[0]) / sr) * 1000))
        self._trim_updating_slider = True
        self.trim_dialog.timeline.set_duration_ms(max_ms)
        self.trim_dialog.timeline.set_range_ms(0, max_ms, emit=False)
        self.trim_dialog.timeline.set_playhead_ms(0, emit=False)
        self._trim_updating_slider = False
        self.trim_dialog.trim_target_label.setText(src.name)
        self.trim_dialog.trim_time_label.setText(f"00:00.00 / {self._format_ms(max_ms)}")
        self.trim_play_pause_btn.setText("Play")

        self.refresh_local()
        for i in range(self.local_list.count()):
            item = self.local_list.item(i)
            if item.text().strip().lower() == src.name.lower():
                self.local_list.setCurrentItem(item)
                break
        self.status_label.setText(f"Trim saved to '{src.name}'.")

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
        hotkeys_need_rebuild = False
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
            for hotkey, mapped_name in list(self.engine.soundboard.hotkeys.items()):
                if mapped_name == sound_name:
                    hotkeys_need_rebuild = True
                    break

        if hotkeys_need_rebuild:
            try:
                self.engine.soundboard.clear_hotkeys()
                self.engine.soundboard.bind_hotkeys_auto()
            except Exception as exc:
                self.status_label.setText(f"Deleted files, but failed to rebuild hotkeys: {exc}")

        self._exit_delete_mode()
        self.refresh_local()
        self.status_label.setText(f"Deleted {deleted} sound(s).")

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
            flags = item.flags()
            if enabled:
                item.setFlags(flags | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setFlags(flags & ~Qt.ItemFlag.ItemIsUserCheckable)

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
        gain = slider_value / 100.0
        self.engine.soundboard.set_volume(gain)
        self.soundboard_volume_label.setText(f"Soundboard Volume: {slider_value}%")

    def toggle_speaker_monitor(self, enabled: bool) -> None:
        self.speaker_monitor_btn.setText("Play To Speaker: On" if enabled else "Play To Speaker: Off")
        self.speaker_monitor_label.setVisible(enabled)
        self.speaker_monitor_slider.setVisible(enabled)
        if not enabled:
            try:
                sd.stop()
            except Exception:
                pass

    def update_speaker_monitor_volume_label(self, slider_value: int) -> None:
        self.speaker_monitor_label.setText(f"Speaker Volume: {slider_value}%")

    def _play_sound_to_speaker_if_enabled(self, sound_name: str) -> None:
        if not self.speaker_monitor_btn.isChecked():
            return
        snd = self.engine.soundboard.sounds.get(sound_name)
        if snd is None:
            return
        gain = self.speaker_monitor_slider.value() / 100.0
        if gain <= 0.0:
            return
        try:
            audio = np.clip(snd.audio * gain, -1.0, 1.0)
            sd.play(audio, samplerate=self.engine.samplerate, blocking=False)
        except Exception as exc:
            self.status_label.setText(f"Speaker playback failed: {exc}")

    def update_preview_volume_label(self, slider_value: int) -> None:
        self.preview_volume_label.setText(f"Preview Volume: {slider_value}%")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.stop_remote_preview(silent=True)
        self.stop_trim_preview(silent=True)
        self.engine.stop()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_local_grid_size()
        if not self._delete_mode:
            self.apply_local_filter()


def run_ui() -> int:
    app = QApplication(sys.argv)
    window = SoundboardWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_ui())
