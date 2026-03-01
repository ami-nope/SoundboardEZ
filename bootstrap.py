"""SoundboardEZ bootstrap installer.

Build with:
    pyinstaller --onefile --noconsole --name SoundboardEZ-Setup --icon assets/app.ico bootstrap.py

Flow:
    1. Show branded UpdateWindow in "install" mode.
    2. Fetch manifest → get version + full package URL.
    3. Stream-download the ZIP while showing progress / speed / ETA.
    4. Extract into %LOCALAPPDATA%\\SoundboardEZ\\.
    5. Create Desktop + Start Menu shortcuts.
    6. Launch SoundboardEZ.exe silently.
    7. Close installer.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from update_ui import UpdateWindow, create_download_worker

# ── constants ──────────────────────────────────────────────────────────────

MANIFEST_URL = "https://soundboardez.up.railway.app/manifest"
INSTALL_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "SoundboardEZ"
EXE_NAME = "SoundboardEZ.exe"
CREATE_NO_WINDOW = 0x08000000


# ── manifest fetch ─────────────────────────────────────────────────────────

def _fetch_manifest() -> dict:
    import requests
    resp = requests.get(MANIFEST_URL, timeout=(8.0, 30.0))
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid manifest.")
    return payload


# ── shortcut creation ──────────────────────────────────────────────────────

def _create_shortcuts(exe_path: Path) -> None:
    """Create Desktop and Start Menu shortcuts for SoundboardEZ.

    Uses PowerShell COM via WScript.Shell (no pywin32 dependency).
    Runs silently with CREATE_NO_WINDOW.
    """
    desktop = Path(os.path.expandvars(r"%USERPROFILE%\Desktop"))
    start_menu = Path(os.path.expandvars(
        r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"
    ))

    icon_path = exe_path.parent / "assets" / "app.ico"
    icon_arg = str(icon_path) if icon_path.exists() else str(exe_path)

    for folder in (desktop, start_menu):
        lnk = folder / "SoundboardEZ.lnk"
        # PowerShell script to create a .lnk via WScript.Shell COM
        ps = (
            '$ws = New-Object -ComObject WScript.Shell; '
            f'$s = $ws.CreateShortcut("{lnk}"); '
            f'$s.TargetPath = "{exe_path}"; '
            f'$s.WorkingDirectory = "{exe_path.parent}"; '
            f'$s.IconLocation = "{icon_arg},0"; '
            '$s.Description = "SoundboardEZ"; '
            '$s.Save()'
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                capture_output=True,
                timeout=15,
                creationflags=CREATE_NO_WINDOW,
            )
        except Exception:
            pass  # shortcuts are best-effort


# ── extract worker ─────────────────────────────────────────────────────────

class _ExtractWorker(QObject):
    finished = pyqtSignal(str)   # extracted root path
    error = pyqtSignal(str)

    def __init__(self, zip_path: Path, dest: Path) -> None:
        super().__init__()
        self._zip = Path(zip_path)
        self._dest = Path(dest)

    def run(self) -> None:
        try:
            if self._dest.exists():
                shutil.rmtree(self._dest, ignore_errors=True)
            self._dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(self._zip, "r") as zf:
                zf.extractall(self._dest)
            # If the extracted content has a single top-level dir, use that
            entries = [p for p in self._dest.iterdir()]
            dirs = [p for p in entries if p.is_dir()]
            files = [p for p in entries if p.is_file()]
            root = dirs[0] if not files and len(dirs) == 1 else self._dest
            self.finished.emit(str(root))
        except Exception as exc:
            self.error.emit(str(exc))


# ── installer logic ───────────────────────────────────────────────────────

class InstallerController:
    """Orchestrates: manifest → download → extract → launch → exit."""

    def __init__(self, app: QApplication) -> None:
        self._app = app
        self._win: UpdateWindow | None = None
        self._dl_thread: QThread | None = None
        self._ex_thread: QThread | None = None
        self._version = ""
        self._full_url = ""
        self._pkg_path = INSTALL_DIR / "update.pkg"

    def start(self) -> int:
        # Fetch manifest synchronously (quick, <1 s typically)
        try:
            manifest = _fetch_manifest()
        except Exception as exc:
            QMessageBox.critical(None, "SoundboardEZ", f"Failed to reach update server:\n{exc}")
            return 1

        self._version = str(manifest.get("version", "")).strip().lstrip("vV")
        full = manifest.get("full")
        if not isinstance(full, dict) or not full.get("url"):
            QMessageBox.critical(None, "SoundboardEZ", "Manifest is missing full package URL.")
            return 1
        self._full_url = str(full["url"]).strip()

        self._win = UpdateWindow(
            mode="install",
            new_version=self._version,
            mandatory=True,
        )
        self._win.retry_requested.connect(self._begin_download)
        QTimer.singleShot(0, self._begin_download)
        result = self._win.exec()
        return 0 if result == QDialog.DialogCode.Accepted else 1

    # ── download phase ─────────────────────────────────────────────────────

    def _begin_download(self) -> None:
        if self._win is None:
            return
        self._win.reset_for_retry()
        worker = create_download_worker(self._full_url, self._pkg_path)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        worker.stage_changed.connect(self._win.set_stage)
        worker.progress.connect(self._win.set_progress)
        worker.speed.connect(self._win.set_speed)
        worker.eta.connect(self._win.set_eta)
        worker.finished.connect(self._on_download_done)
        worker.error.connect(self._on_download_error)

        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._dl_thread = thread
        thread.start()

    def _on_download_error(self, msg: str) -> None:
        if self._win:
            self._win.set_error(f"Download failed: {msg}")

    def _on_download_done(self, local_path: str) -> None:
        self._begin_extract(Path(local_path))

    # ── extract phase ──────────────────────────────────────────────────────

    def _begin_extract(self, pkg: Path) -> None:
        if self._win:
            self._win.set_stage("extracting")

        worker = _ExtractWorker(pkg, INSTALL_DIR)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        worker.finished.connect(self._on_extract_done)
        worker.error.connect(self._on_extract_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._ex_thread = thread
        thread.start()

    def _on_extract_error(self, msg: str) -> None:
        if self._win:
            self._win.set_error(f"Extraction failed: {msg}")

    def _on_extract_done(self, root_path: str) -> None:
        root = Path(root_path)
        exe = root / EXE_NAME
        if not exe.exists():
            # Try install dir root
            exe = INSTALL_DIR / EXE_NAME
        if not exe.exists():
            if self._win:
                self._win.set_error(f"{EXE_NAME} not found after extraction.")
            return

        # Clean up the downloaded zip
        try:
            self._pkg_path.unlink(missing_ok=True)
        except Exception:
            pass

        # Create Desktop + Start Menu shortcuts
        _create_shortcuts(exe)

        if self._win:
            self._win.set_stage("finalizing")
            self._win.set_complete()

        # Launch the installed app silently (no console window)
        try:
            flags = 0
            if os.name == "nt":
                flags = CREATE_NO_WINDOW
            subprocess.Popen(
                [str(exe)],
                creationflags=flags,
                close_fds=True,
                cwd=str(exe.parent),
            )
        except Exception:
            pass

        # Close installer after a brief moment
        if self._win:
            QTimer.singleShot(600, self._win.accept)


# ── entry point ────────────────────────────────────────────────────────────

def main() -> int:
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    app.setQuitOnLastWindowClosed(True)
    controller = InstallerController(app)
    return controller.start()


if __name__ == "__main__":
    raise SystemExit(main())
