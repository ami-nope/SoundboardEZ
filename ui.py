from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
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
    QPoint,
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
from PyQt6.QtGui import QColor, QBrush, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
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
    QScrollArea,
    QSpacerItem,
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

        rect = option.rect.adjusted(8, 8, -8, -8)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        checked = index.data(int(Qt.ItemDataRole.CheckStateRole)) == Qt.CheckState.Checked

        base_color = QColor("#334155")
        if hovered:
            base_color = QColor("#3f4f66")
        if selected:
            base_color = QColor("#425a76")

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(2, 8, 23, 76))
        painter.drawRoundedRect(rect.adjusted(0, 1, 0, 1), 14, 14)

        border_color = QColor("#475569")
        border_width = 2 if (selected or hovered) else 1
        if hovered and not selected:
            border_color = QColor(56, 189, 248, 125)
        if selected:
            border_color = QColor(56, 189, 248, 225)

        painter.setBrush(base_color)
        pen = QPen(border_color, border_width)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 14, 14)

        if hovered or selected:
            glow_alpha = 28 if hovered else 42
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(56, 189, 248, glow_alpha))
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 13, 13)

        text = str(index.data(int(Qt.ItemDataRole.DisplayRole)) or "")
        text_rect = rect.adjusted(10, 0, -10, 0)
        text = option.fontMetrics.elidedText(text, Qt.TextElideMode.ElideRight, text_rect.width())
        painter.setPen(QColor("#e2e8f0"))
        painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignCenter), text)

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
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._dragging = False
        self._drag_offset = QPoint()

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


class SoundboardWindow(QMainWindow):
    DEFAULT_KEYS = list("1234567890qwertyuiopasdfghjklzxcvbnm")

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SoundboardEZ")
        self.resize(920, 520)
        self.setMinimumSize(920, 520)

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
        self._soundboard_initialized = False

        self.engine = AudioEngine(
            samplerate=48000,
            blocksize=512,
            input_channels=1,
            output_channels=1,
            sounds_dir=str(self.sounds_dir),
        )
        self._engine_thread = threading.Thread(target=self._run_engine, daemon=True)
        self._engine_thread.start()

        central = QWidget()
        central.setObjectName("Root")
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
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
        self.app_subtitle = QLabel("Virtual Mic Mixer")
        self.app_subtitle.setObjectName("AppSubtitle")
        title_col.addWidget(self.app_title)
        title_col.addWidget(self.app_subtitle)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusPill")
        self.route_status_label = QLabel("Hosted On: detecting... | Mic: detecting...")
        self.route_status_label.setObjectName("RoutePill")
        self.route_status_label.setMinimumWidth(0)
        self.route_status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        state_col = QVBoxLayout()
        state_col.setContentsMargins(0, 0, 0, 0)
        state_col.setSpacing(8)
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
        self.import_search_input.setPlaceholderText("Search myinstants sounds")
        self.import_search_input.setClearButtonEnabled(True)
        self.import_search_btn = QPushButton("Search")
        self.import_search_btn.setProperty("variant", "primary")
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
        self.mic_volume_label = QLabel("Mic Volume: 50%")
        self.mic_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.mic_volume_slider.setRange(0, 100)
        self.mic_volume_slider.setValue(50)
        self.remote_feed_list = SmoothListWidget(slow_factor=0.5)
        self.remote_feed_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.remote_feed_list.setSpacing(16)
        self.remote_feed_list.setMinimumHeight(120)

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
        importer_search_row.addWidget(self.import_search_btn)

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

        importer_mic_row = QHBoxLayout()
        importer_mic_row.setContentsMargins(0, 0, 0, 0)
        importer_mic_row.setSpacing(16)
        importer_mic_row.addWidget(self.mic_volume_label)
        importer_mic_row.addWidget(self.mic_volume_slider, 1)

        left_layout.addLayout(importer_header_row)
        left_layout.addLayout(importer_search_row)
        left_layout.addLayout(importer_actions_row)
        left_layout.addLayout(importer_preview_row)
        left_layout.addLayout(importer_mic_row)
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
        right_layout.setContentsMargins(16, 16, 16, 16)
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
        main_layout = QVBoxLayout(self.soundboard_main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(16)
        self.soundboard_volume_label = QLabel("Soundboard Volume: 100%")
        self.soundboard_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.soundboard_volume_slider.setRange(0, 200)
        self.soundboard_volume_slider.setValue(100)
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
        self.device_label = QLabel("Audio Route")
        self.device_label.setObjectName("HintLabel")
        self.mic_device_label = QLabel("Mic Input")
        self.mic_device_combo = QComboBox()
        self.mic_device_combo.setObjectName("RouteCombo")
        self.output_device_label = QLabel("Mix Output")
        self.output_device_combo = QComboBox()
        self.output_device_combo.setObjectName("RouteCombo")
        self.refresh_devices_btn = QPushButton("Refresh Devices")
        self.refresh_devices_btn.setProperty("variant", "slate")
        self.apply_devices_btn = QPushButton("Apply Route")
        self.apply_devices_btn.setProperty("variant", "primary")
        self.mic_noise_suppression_btn = QPushButton("Mic Noise Suppression: Off")
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
        self.local_search_input.setPlaceholderText("Search imported sounds")
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
        self.local_list.setSpacing(24)
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
        self.sidebar_scroll.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.sidebar_scroll.setMinimumWidth(sidebar_w + 10)
        self.sidebar_scroll.setMaximumWidth(sidebar_w + 10)
        self.sidebar_scroll.setWidget(self.soundboard_sidebar)

        local_header = QHBoxLayout()
        local_header.setSpacing(16)
        local_header.addWidget(self.local_title)
        local_header.addStretch(1)
        local_header.addWidget(self.local_search_input, 0)
        main_layout.addLayout(local_header)
        main_layout.addWidget(self.local_list, 1)

        right_layout.addWidget(self.soundboard_main, 1)
        main_column_layout.addWidget(self.right_group, 1)
        root_layout.addWidget(self.sidebar_scroll)
        root_layout.addWidget(self.main_content, 1)

        self.importer_window = FramelessImporterDialog(self)
        self.importer_window.setObjectName("ImporterWindow")
        self.importer_window.setWindowTitle("Importer - SoundboardEZ")
        self.importer_window.setModal(False)
        self.importer_window.setMinimumSize(680, 520)
        self.importer_window.resize(760, 620)
        self.importer_shell = QWidget()
        self.importer_shell.setObjectName("ImporterShell")
        importer_shadow = QGraphicsDropShadowEffect(self.importer_shell)
        importer_shadow.setBlurRadius(38)
        importer_shadow.setOffset(0, 14)
        importer_shadow.setColor(QColor(4, 11, 24, 180))
        self.importer_shell.setGraphicsEffect(importer_shadow)
        importer_shell_layout = QVBoxLayout(self.importer_shell)
        importer_shell_layout.setContentsMargins(20, 20, 20, 20)
        importer_shell_layout.setSpacing(0)
        importer_shell_layout.addWidget(self.left_group)
        importer_window_layout = QVBoxLayout(self.importer_window)
        importer_window_layout.setContentsMargins(8, 8, 8, 8)
        importer_window_layout.setSpacing(0)
        importer_window_layout.addWidget(self.importer_shell)
        self.importer_window.finished.connect(lambda _=0: self._set_importer_visible(False))

        self.import_search_btn.clicked.connect(self.apply_import_search)
        self.close_importer_btn.clicked.connect(self.close_importer_panel)
        self.import_search_input.returnPressed.connect(self.apply_import_search)
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
        self._importer_loaded_once = False
        self._set_importer_visible(False)
        self.refresh_audio_devices()
        self.refresh_local()
        self.update_preview_volume_label(self.preview_volume_slider.value())
        self.update_mic_volume_label(self.mic_volume_slider.value())
        self.update_soundboard_volume(self.soundboard_volume_slider.value())
        self.update_speaker_monitor_volume_label(self.speaker_monitor_slider.value())
        self.engine.set_noise_suppression_enabled(False)
        self._sync_mic_noise_suppression_button()
        self._update_route_status()
        self._run_entrance_animation()

    def _run_engine(self) -> None:
        try:
            if not self._soundboard_initialized:
                mapping = self.engine.setup_soundboard(auto_hotkeys=True)
                self._soundboard_initialized = True
                if mapping:
                    print("Initial hotkeys:")
                    for key, name in mapping.items():
                        print(f"  {key} -> {name}")
            self.engine.start()
        except Exception as exc:
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

    def _restart_audio_engine(self) -> None:
        self.engine.stop()
        if self._engine_thread.is_alive():
            self._engine_thread.join(timeout=2.5)
        self._engine_thread = threading.Thread(target=self._run_engine, daemon=True)
        self._engine_thread.start()

    def apply_audio_route(self) -> None:
        in_dev = self._coerce_device_data(self.mic_device_combo.currentData())
        out_dev = self._coerce_device_data(self.output_device_combo.currentData())
        self.engine.input_device = in_dev
        self.engine.output_device = out_dev
        try:
            self._restart_audio_engine()
            self.status_label.setText("Audio route applied.")
            self._update_route_status()
        except Exception as exc:
            self.status_label.setText(f"Failed to apply route: {exc}")

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
            self.import_search_btn,
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
        self.mic_volume_slider.setMinimumWidth(220)
        self.mic_device_combo.setMinimumHeight(32)
        self.mic_device_combo.setMinimumWidth(0)
        self.mic_device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.output_device_combo.setMinimumHeight(32)
        self.output_device_combo.setMinimumWidth(0)
        self.output_device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _apply_modern_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget#Root {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0a111d,
                    stop:1 #0f1b2f);
                color: #e7edf7;
                font-family: "SF Pro Text", "Segoe UI Variable Text", "Segoe UI", sans-serif;
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
                background: rgba(16, 25, 39, 238);
                border: 1px solid rgba(148, 163, 184, 38);
                border-radius: 28px;
            }
            QFrame#TopBar {
                background: rgba(37, 53, 74, 210);
                border: 1px solid rgba(148, 163, 184, 36);
                border-radius: 20px;
            }
            QLabel#LogoBadge {
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
                border-radius: 16px;
                background: rgba(12, 20, 34, 188);
                border: 1px solid #7dd3fc;
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
                color: #9eb2cc;
                font-size: 15px;
                font-weight: 500;
            }
            QGroupBox#MainCard {
                background: rgba(30, 42, 60, 205);
                border: 1px solid rgba(148, 163, 184, 32);
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
            QWidget#SideCard {
                background: rgba(40, 56, 78, 176);
                border: 1px solid rgba(148, 163, 184, 30);
                border-radius: 20px;
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
                background: rgba(31, 46, 67, 186);
                border: 1px solid rgba(148, 163, 184, 30);
                border-radius: 20px;
                padding: 10px;
            }
            QLabel#SectionTitle {
                font-size: 22px;
                font-weight: 600;
                color: #edf3fb;
                letter-spacing: 0.2px;
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
                selection-background-color: #45c8ff;
                color: #e7edf7;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #62d6ff;
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
                background: rgba(62, 79, 104, 220);
                color: #eaf2fb;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background: rgba(76, 97, 126, 230);
                border-color: rgba(96, 214, 255, 190);
            }
            QPushButton:pressed {
                background: rgba(56, 74, 98, 235);
            }
            QPushButton:checked,
            QPushButton[active="true"] {
                background: #0f6f98;
                border-color: #62d6ff;
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
            QListWidget#LocalSoundGrid {
                background: rgba(14, 26, 45, 226);
                border: 1px solid rgba(148, 163, 184, 36);
                border-radius: 18px;
                padding: 10px;
            }
            QListWidget#LocalSoundGrid::item {
                border: none;
                color: #edf4fc;
            }
            QListWidget#LocalSoundGrid::item:selected {
                background: transparent;
                color: #edf4fc;
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
                background: #45c8ff;
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
            """
        )

    def _apply_button_motion(self) -> None:
        # Keep interaction light-weight in the minimal theme.
        return

    def _run_entrance_animation(self) -> None:
        sequence = [
            (self.top_bar, 0),
            (self.sidebar_scroll, 70),
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
            self.importer_window.hide()
        self.toggle_importer_btn.setText("Close Importer" if visible else "Open Importer")
        if visible and not self._importer_loaded_once:
            self.fetch_sounds()
            self._importer_loaded_once = True

    def toggle_importer_panel(self) -> None:
        self._set_importer_visible(not self.importer_window.isVisible())

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
            friendly = self._friendly_fetch_error(message)
            self.status_label.setText(friendly)
            print(f"Feed load error: {message}")

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
        bar = self.remote_feed_list.verticalScrollBar()
        if bar.maximum() <= 0:
            return
        if value >= bar.maximum() - 80:
            self._load_next_feed_page()

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
        play_btn.setMinimumHeight(32)
        play_btn.setMinimumWidth(78)
        play_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        import_btn = QPushButton("Import")
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
        feed_button_style = """
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
        play_btn.setProperty("active", False)
        import_btn.setProperty("active", False)
        play_btn.setStyleSheet(feed_button_style)
        import_btn.setStyleSheet(feed_button_style)

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
            btn.setProperty("active", text.strip().lower() == "stop")
            self._refresh_dynamic_button_style(btn)

    def _reset_feed_play_buttons(self) -> None:
        for btn in self._feed_play_buttons.values():
            btn.setText("Play")
            btn.setEnabled(True)
            btn.setProperty("active", False)
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
        tile_w, tile_h, _ = self._compute_local_tile_metrics()
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
        view_w = max(260, self.local_list.viewport().width())
        min_tile_w = max(108, int(78 * PHI))
        max_tile_w = 214
        cols = max(1, min(6, int((view_w + spacing) / (min_tile_w + spacing))))
        tile_w = int((view_w - (spacing * (cols + 1))) / cols)
        tile_w = max(min_tile_w, min(max_tile_w, tile_w))
        tile_h = max(42, int(tile_w / (PHI * 1.18)))
        return tile_w, tile_h, spacing

    def _update_local_grid_size(self) -> None:
        tile_w, tile_h, spacing = self._compute_local_tile_metrics()
        self.local_list.setGridSize(QSize(tile_w + spacing, tile_h + 12))

    def _refresh_local_item_size_hints(self) -> None:
        tile_w, tile_h, _ = self._compute_local_tile_metrics()
        size = QSize(tile_w, tile_h)
        for idx in range(self.local_list.count()):
            item = self.local_list.item(idx)
            if item is not None:
                item.setSizeHint(size)

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
        if self._delete_mode:
            return
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

    def update_mic_volume_label(self, slider_value: int) -> None:
        value = max(0, min(100, int(slider_value)))
        self.mic_volume_label.setText(f"Mic Volume: {value}%")
        self.engine.set_mic_input_gain(value / 100.0)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.stop_remote_preview(silent=True)
        self.stop_trim_preview(silent=True)
        self.engine.stop()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_local_grid_size()
        self._refresh_local_item_size_hints()


def run_ui() -> int:
    app = QApplication(sys.argv)
    window = SoundboardWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_ui())
