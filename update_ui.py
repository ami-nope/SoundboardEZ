"""Reusable branded UpdateWindow for SoundboardEZ.

Used by both the bootstrap installer (``bootstrap.py``) and the in-app
updater.  Supports two modes — *install* and *update* — which only change
the displayed text; the layout and branding stay identical.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PyQt6.QtCore import (
    QObject,
    QSize,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ── constants ──────────────────────────────────────────────────────────────

_LOGO_SIZE = 64
_WIN_MIN_W = 460
_WIN_MIN_H = 280

_BG = "rgba(16, 24, 40, 248)"
_BORDER = "rgba(148, 163, 184, 44)"
_TEXT_PRIMARY = "#edf3fb"
_TEXT_SECONDARY = "#c5d4e8"
_TEXT_MUTED = "#8fa5c1"
_ACCENT = "#59d7ff"
_ACCENT_GLOW = "rgba(89, 215, 255, 45)"
_PROGRESS_BG = "rgba(11, 21, 36, 220)"
_BTN_BG = "rgba(64, 82, 108, 220)"
_BTN_HOVER = "rgba(81, 101, 130, 230)"
_BTN_PRESS = "rgba(58, 77, 102, 235)"
_BTN_DISABLED_BG = "rgba(27, 39, 58, 220)"
_BTN_DISABLED_TEXT = "#8fa5c1"
_ERR_COLOR = "#ff6b6b"


def _resolve_icon_path() -> str | None:
    """Return the absolute path to ``assets/app.ico`` if it exists."""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else Path(sys.executable).parent
    else:
        base = Path(__file__).resolve().parent
    candidates = [base / "assets" / "app.ico", base / "app.ico"]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _make_logo_pixmap(size: int = _LOGO_SIZE) -> QPixmap:
    """Create a simple branded circle logo (fallback when no icon file)."""
    pm = QPixmap(size, size)
    pm.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(89, 215, 255, 40))
    painter.setPen(QPen(QColor(89, 215, 255, 140), 2))
    painter.drawEllipse(4, 4, size - 8, size - 8)
    font = QFont("Segoe UI", int(size * 0.32), QFont.Weight.Bold)
    painter.setFont(font)
    painter.setPen(QColor(_TEXT_PRIMARY))
    painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "S")
    painter.end()
    return pm


# ── download worker (runs in QThread) ──────────────────────────────────────

class _DownloadWorker(QObject):
    """Streams a file and emits byte-level progress."""

    progress = pyqtSignal(int, int)          # downloaded_bytes, total_bytes (-1 if unknown)
    speed = pyqtSignal(float)                # bytes / second
    eta = pyqtSignal(float)                  # seconds remaining (-1 if unknown)
    stage_changed = pyqtSignal(str)          # "connecting" | "downloading" | "extracting" | "finalizing"
    finished = pyqtSignal(str)               # local path on success
    error = pyqtSignal(str)                  # message on failure

    def __init__(self, url: str, dest: Path) -> None:
        super().__init__()
        self._url = str(url)
        self._dest = Path(dest)

    def run(self) -> None:
        import requests

        self.stage_changed.emit("connecting")
        partial = self._dest.parent / (self._dest.name + ".part")
        self._dest.parent.mkdir(parents=True, exist_ok=True)

        for p in (self._dest, partial):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        downloaded = 0
        total: int = -1
        t0 = time.monotonic()
        last_speed_time = t0
        last_speed_bytes = 0

        try:
            with requests.get(self._url, stream=True, timeout=(8.0, 120.0)) as resp:
                resp.raise_for_status()
                cl = str(resp.headers.get("content-length", "")).strip()
                if cl.isdigit():
                    total = int(cl)

                self.stage_changed.emit("downloading")
                self.progress.emit(0, total)

                with partial.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=512 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        self.progress.emit(downloaded, total)

                        now = time.monotonic()
                        dt = now - last_speed_time
                        if dt >= 0.4:
                            spd = (downloaded - last_speed_bytes) / dt
                            self.speed.emit(spd)
                            if spd > 0 and total > 0:
                                self.eta.emit((total - downloaded) / spd)
                            else:
                                self.eta.emit(-1)
                            last_speed_time = now
                            last_speed_bytes = downloaded

            partial.replace(self._dest)
            self.progress.emit(total if total > 0 else downloaded, total if total > 0 else downloaded)
            self.finished.emit(str(self._dest))
        except Exception as exc:
            try:
                partial.unlink(missing_ok=True)
            except Exception:
                pass
            self.error.emit(str(exc))


# ── UpdateWindow ───────────────────────────────────────────────────────────

class UpdateWindow(QDialog):
    """Branded progress dialog for installing or updating SoundboardEZ.

    Parameters
    ----------
    mode : ``"install"`` or ``"update"``
    current_version : shown only when *mode* is ``"update"``
    new_version : target version string
    mandatory : when True, the cancel/skip button is disabled
    parent : optional parent widget
    """

    update_complete = pyqtSignal()       # emitted when handoff is about to happen
    retry_requested = pyqtSignal()       # emitted when user clicks Retry after error
    cancel_requested = pyqtSignal()      # emitted on cancel/skip

    def __init__(
        self,
        mode: str = "update",
        current_version: str = "",
        new_version: str = "",
        mandatory: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._mandatory = bool(mandatory)
        self._busy = False
        self._allow_close = not self._mandatory

        # ── window flags ──
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Dialog
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumSize(_WIN_MIN_W, _WIN_MIN_H)
        self.setModal(True)

        icon_path = _resolve_icon_path()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("SoundboardEZ")

        # ── inner card (provides the rounded background) ──
        self._card = QWidget(self)
        self._card.setObjectName("UpdateCard")

        card_shadow = QGraphicsDropShadowEffect(self._card)
        card_shadow.setBlurRadius(32)
        card_shadow.setOffset(0, 4)
        card_shadow.setColor(QColor(0, 0, 0, 120))
        self._card.setGraphicsEffect(card_shadow)

        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(28, 24, 28, 22)
        card_layout.setSpacing(10)

        # ── logo ──
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if icon_path:
            pm = QPixmap(icon_path).scaled(
                _LOGO_SIZE, _LOGO_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            pm = _make_logo_pixmap(_LOGO_SIZE)
        logo_label.setPixmap(pm)
        card_layout.addWidget(logo_label)

        # ── title ──
        if mode == "install":
            title_text = "Installing SoundboardEZ"
        else:
            title_text = "Updating SoundboardEZ"
        self._title = QLabel(title_text)
        self._title.setObjectName("UWTitle")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self._title)

        # ── subtitle (version info) ──
        if mode == "update" and current_version:
            sub_text = f"v{current_version}  →  v{new_version}"
        else:
            sub_text = f"v{new_version}" if new_version else ""
        self._subtitle = QLabel(sub_text)
        self._subtitle.setObjectName("UWSubtitle")
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self._subtitle)

        # ── progress bar ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("UWProgress")
        self.progress_bar.setRange(0, 0)  # indeterminate by default
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        card_layout.addWidget(self.progress_bar)

        # ── stats row: percentage | speed | ETA ──
        stats_row = QHBoxLayout()
        stats_row.setContentsMargins(0, 0, 0, 0)
        stats_row.setSpacing(0)
        self._pct_label = QLabel("0 %")
        self._pct_label.setObjectName("UWStats")
        self._speed_label = QLabel("")
        self._speed_label.setObjectName("UWStats")
        self._speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._eta_label = QLabel("")
        self._eta_label.setObjectName("UWStats")
        self._eta_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        stats_row.addWidget(self._pct_label)
        stats_row.addStretch(1)
        stats_row.addWidget(self._speed_label)
        stats_row.addStretch(1)
        stats_row.addWidget(self._eta_label)
        card_layout.addLayout(stats_row)

        # ── status label ──
        self._status = QLabel("Connecting…")
        self._status.setObjectName("UWStatus")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self._status)

        # ── buttons ──
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 6, 0, 0)
        btn_row.setSpacing(10)
        btn_row.addStretch(1)
        self._cancel_btn = QPushButton("Cancel" if mode == "install" else "Skip")
        self._cancel_btn.setObjectName("UWBtnSecondary")
        self._cancel_btn.setFixedHeight(32)
        self._cancel_btn.setMinimumWidth(80)
        self._retry_btn = QPushButton("Retry")
        self._retry_btn.setObjectName("UWBtnPrimary")
        self._retry_btn.setFixedHeight(32)
        self._retry_btn.setMinimumWidth(80)
        self._retry_btn.setVisible(False)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._retry_btn)
        card_layout.addLayout(btn_row)

        if self._mandatory:
            self._cancel_btn.setEnabled(False)

        self._cancel_btn.clicked.connect(self._on_cancel)
        self._retry_btn.clicked.connect(self._on_retry)

        # ── outer layout (provides margin around the card for the shadow) ──
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.addWidget(self._card)

        self._apply_stylesheet()

    # ── stylesheet ─────────────────────────────────────────────────────────

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(f"""
            QWidget#UpdateCard {{
                background: {_BG};
                border: 1px solid {_BORDER};
                border-radius: 16px;
            }}
            QLabel#UWTitle {{
                color: {_TEXT_PRIMARY};
                font-size: 18px;
                font-weight: 700;
            }}
            QLabel#UWSubtitle {{
                color: {_TEXT_SECONDARY};
                font-size: 13px;
            }}
            QLabel#UWStatus {{
                color: {_TEXT_SECONDARY};
                font-size: 12px;
                font-weight: 500;
            }}
            QLabel#UWStats {{
                color: {_TEXT_MUTED};
                font-size: 11px;
            }}
            QProgressBar#UWProgress {{
                background: {_PROGRESS_BG};
                border: 1px solid {_BORDER};
                border-radius: 5px;
            }}
            QProgressBar#UWProgress::chunk {{
                background: {_ACCENT};
                border-radius: 4px;
            }}
            QPushButton#UWBtnSecondary {{
                background: {_BTN_BG};
                border: 1px solid {_BORDER};
                border-radius: 8px;
                color: {_TEXT_PRIMARY};
                font-size: 12px;
                font-weight: 600;
                padding: 0 14px;
            }}
            QPushButton#UWBtnSecondary:hover {{
                background: {_BTN_HOVER};
            }}
            QPushButton#UWBtnSecondary:pressed {{
                background: {_BTN_PRESS};
            }}
            QPushButton#UWBtnSecondary:disabled {{
                background: {_BTN_DISABLED_BG};
                color: {_BTN_DISABLED_TEXT};
            }}
            QPushButton#UWBtnPrimary {{
                background: {_ACCENT};
                border: none;
                border-radius: 8px;
                color: #0b1525;
                font-size: 12px;
                font-weight: 700;
                padding: 0 14px;
            }}
            QPushButton#UWBtnPrimary:hover {{
                background: #7de3ff;
            }}
            QPushButton#UWBtnPrimary:pressed {{
                background: #40c9f0;
            }}
        """)

    # ── public API used by callers ─────────────────────────────────────────

    def set_stage(self, stage: str) -> None:
        labels = {
            "connecting": "Connecting…",
            "downloading": "Downloading…",
            "extracting": "Extracting…",
            "finalizing": "Finalizing…",
            "installing": "Applying update and relaunching…",
        }
        self._status.setText(labels.get(stage, stage))

    def set_progress(self, downloaded: int, total: int) -> None:
        self._busy = True
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(max(0, min(total, downloaded)))
            pct = int(downloaded * 100 / total)
            self._pct_label.setText(f"{pct} %")
        else:
            self.progress_bar.setRange(0, 0)
            self._pct_label.setText("")

    def set_speed(self, bytes_per_sec: float) -> None:
        if bytes_per_sec <= 0:
            self._speed_label.setText("")
            return
        mb = bytes_per_sec / (1024 * 1024)
        if mb >= 1.0:
            self._speed_label.setText(f"{mb:.1f} MB/s")
        else:
            kb = bytes_per_sec / 1024
            self._speed_label.setText(f"{kb:.0f} KB/s")

    def set_eta(self, seconds: float) -> None:
        if seconds < 0:
            self._eta_label.setText("")
            return
        if seconds < 60:
            self._eta_label.setText(f"{int(seconds)}s remaining")
        else:
            m = int(seconds) // 60
            s = int(seconds) % 60
            self._eta_label.setText(f"{m}m {s}s remaining")

    def set_error(self, message: str) -> None:
        self._busy = False
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self._pct_label.setText("")
        self._speed_label.setText("")
        self._eta_label.setText("")
        short = message[:200] if len(message) > 200 else message
        self._status.setText(short)
        self._status.setStyleSheet(f"color: {_ERR_COLOR}; font-size: 12px; font-weight: 500;")
        self._retry_btn.setVisible(True)
        if not self._mandatory:
            self._cancel_btn.setEnabled(True)

    def set_complete(self) -> None:
        self._busy = False
        self._allow_close = True
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self._pct_label.setText("100 %")
        self._speed_label.setText("")
        self._eta_label.setText("")
        self._status.setText("Done!")
        self._status.setStyleSheet(f"color: {_ACCENT}; font-size: 12px; font-weight: 600;")

    def reset_for_retry(self) -> None:
        """Reset UI state so a new download attempt can begin."""
        self._status.setStyleSheet("")  # reset colour
        self._retry_btn.setVisible(False)
        self._cancel_btn.setEnabled(not self._mandatory)
        self.progress_bar.setRange(0, 0)
        self._pct_label.setText("0 %")
        self._speed_label.setText("")
        self._eta_label.setText("")
        self._status.setText("Connecting…")

    # ── internal ───────────────────────────────────────────────────────────

    def _on_cancel(self) -> None:
        self.cancel_requested.emit()
        if not self._mandatory:
            self._allow_close = True
            self.reject()

    def _on_retry(self) -> None:
        self.reset_for_retry()
        self.retry_requested.emit()

    def reject(self) -> None:  # type: ignore[override]
        if self._mandatory and not self._allow_close:
            return
        super().reject()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._busy or (self._mandatory and not self._allow_close):
            event.ignore()
            return
        super().closeEvent(event)

    # ── frameless drag ─────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if hasattr(self, "_drag_pos") and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()


# ── Convenience: DownloadWorker getter ─────────────────────────────────────

def create_download_worker(url: str, dest: Path) -> _DownloadWorker:
    """Create a ``_DownloadWorker`` ready to be moved to a QThread."""
    return _DownloadWorker(url, dest)
