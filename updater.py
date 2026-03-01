"""SoundboardEZ updater — delta patches, full replacement, rollback & crash recovery.

This module is imported by both the main UI process (to kick off downloads) and
the ``--apply-update`` child process (to perform the actual file replacement
while the original exe is not running).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import requests


HTTP_TIMEOUT = (8.0, 120.0)
DEFAULT_EXE_NAME = "SoundboardEZ.exe"
BACKUP_DIR_NAME = "_backup"
FLAG_FILE_NAME = "update_in_progress.flag"


# ── helpers ────────────────────────────────────────────────────────────────

def _install_dir() -> Path:
    """Resolve the application install directory (the folder containing the exe)."""
    return Path(sys.executable).resolve().parent


def _flag_path(install_dir: Path | None = None) -> Path:
    return (install_dir or _install_dir()) / FLAG_FILE_NAME


def _backup_dir(install_dir: Path | None = None) -> Path:
    return (install_dir or _install_dir()) / BACKUP_DIR_NAME


# ── download ───────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, progress_cb=None) -> Path:
    """Stream-download *url* → *.part*, then atomically rename to *dest*."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.parent / (dest.name + ".part")

    for p in (dest, partial):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    downloaded = 0
    total: int | None = None

    with requests.get(str(url), stream=True, timeout=HTTP_TIMEOUT) as resp:
        resp.raise_for_status()
        cl = str(resp.headers.get("content-length", "")).strip()
        if cl.isdigit():
            total = int(cl)
        if callable(progress_cb):
            progress_cb(0, total)

        with partial.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=512 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if callable(progress_cb):
                    progress_cb(downloaded, total)

    partial.replace(dest)
    if callable(progress_cb):
        progress_cb(total or downloaded, total or downloaded)
    return dest


# ── delta download (multiple files) ───────────────────────────────────────

def download_delta_files(
    delta_files: list[tuple[str, str]],
    temp_dir: Path,
    progress_cb=None,
) -> list[tuple[str, Path]]:
    """Download every delta file into *temp_dir*, preserving relative paths.

    *delta_files* is a list of ``(relative_path, url)`` pairs.
    Returns ``[(relative_path, local_path), ...]``.
    """
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    total_files = len(delta_files)
    results: list[tuple[str, Path]] = []

    for idx, (rel_path, url) in enumerate(delta_files):
        local = temp_dir / rel_path
        local.parent.mkdir(parents=True, exist_ok=True)

        def _file_progress(downloaded: int, total: int | None) -> None:
            if callable(progress_cb):
                # Emit an aggregate fraction: file-index / total-files is a rough %
                file_frac = (idx + (downloaded / max(total or 1, 1))) / total_files
                progress_cb(int(file_frac * 100), 100)

        download_file(url, local, progress_cb=_file_progress)
        results.append((rel_path, local))

    return results


# ── full-update extraction ─────────────────────────────────────────────────

def extract_full_package(pkg_path: Path, extract_dir: Path) -> Path:
    """Extract a full ZIP into *extract_dir*.  Returns root of extracted tree."""
    pkg = Path(pkg_path)
    target = Path(extract_dir)

    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(pkg, "r") as zf:
            zf.extractall(target)
    except zipfile.BadZipFile:
        sig = b""
        try:
            with pkg.open("rb") as fh:
                sig = fh.read(2)
        except Exception:
            pass
        if sig == b"MZ":
            exe_name = (
                Path(sys.executable).name
                if getattr(sys, "frozen", False)
                else DEFAULT_EXE_NAME
            )
            shutil.copy2(pkg, target / exe_name)
            return target
        raise RuntimeError("Downloaded file is not a valid ZIP or EXE.")

    entries = [p for p in target.iterdir() if p.exists()]
    if not entries:
        raise RuntimeError("Extracted package is empty.")

    top_files = [p for p in entries if p.is_file()]
    top_dirs = [p for p in entries if p.is_dir()]
    if not top_files and len(top_dirs) == 1:
        return top_dirs[0]
    return target


# ── backup & rollback ─────────────────────────────────────────────────────

def _create_backup(install_dir: Path, relative_paths: list[str]) -> None:
    """Copy originals into ``_backup/`` before overwriting."""
    backup = _backup_dir(install_dir)
    backup.mkdir(parents=True, exist_ok=True)
    for rel in relative_paths:
        src = install_dir / rel
        if not src.exists():
            continue
        dst = backup / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


def rollback_from_backup(install_dir: Path | None = None) -> bool:
    """Restore files from ``_backup/`` and remove the flag.  Returns True on success."""
    root = install_dir or _install_dir()
    backup = _backup_dir(root)
    if not backup.exists():
        return False

    for item in backup.rglob("*"):
        if not item.is_file():
            continue
        rel = item.relative_to(backup)
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(item, dest)
        except Exception:
            pass

    # Clean up
    shutil.rmtree(backup, ignore_errors=True)
    try:
        _flag_path(root).unlink(missing_ok=True)
    except Exception:
        pass
    return True


def check_and_rollback_on_startup() -> bool:
    """If an interrupted update is detected, roll back and return True."""
    root = _install_dir()
    flag = _flag_path(root)
    if not flag.exists():
        return False
    print("[updater] Interrupted update detected – rolling back from _backup")
    return rollback_from_backup(root)


def clear_update_flag() -> None:
    """Called after a successful launch to remove the in-progress flag."""
    try:
        flag = _flag_path()
        flag.unlink(missing_ok=True)
    except Exception:
        pass
    # Also clean up the backup directory on successful launch
    try:
        backup = _backup_dir()
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        pass


# ── write flag ─────────────────────────────────────────────────────────────

def _write_flag(install_dir: Path, update_info_json: str) -> None:
    """Drop the ``update_in_progress.flag`` containing metadata."""
    flag = _flag_path(install_dir)
    flag.write_text(update_info_json, encoding="utf-8")


# ── delta apply ────────────────────────────────────────────────────────────

def _apply_delta(
    downloaded: list[tuple[str, Path]],
    install_dir: Path,
    exe_name: str,
) -> None:
    """Replace individual files in *install_dir* from downloaded delta files."""
    rel_paths = [r for r, _ in downloaded]
    _create_backup(install_dir, rel_paths)

    _write_flag(install_dir, json.dumps({"mode": "delta", "files": rel_paths}))

    for rel_path, local_file in downloaded:
        dest = install_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        # If the target is the running exe, rename it first
        if dest.name.lower() == exe_name.lower() and dest.exists():
            old = dest.parent / (dest.name + ".old")
            try:
                if old.exists():
                    old.unlink()
            except Exception:
                pass
            try:
                dest.rename(old)
            except Exception:
                pass

        try:
            shutil.copy2(local_file, dest)
        except PermissionError:
            # On Windows the exe might still be locked briefly; retry once
            time.sleep(1.0)
            shutil.copy2(local_file, dest)


# ── full apply ─────────────────────────────────────────────────────────────

def _apply_full(
    source_dir: Path,
    install_dir: Path,
    exe_name: str,
) -> None:
    """Replace the entire install directory from the extracted full package.

    User config (``app_state.json`` in ``%APPDATA%``) is NOT in install_dir
    so it survives automatically.
    """
    # Back up all existing files
    existing_files = []
    for item in install_dir.rglob("*"):
        if not item.is_file():
            continue
        rel = str(item.relative_to(install_dir))
        if rel.startswith(BACKUP_DIR_NAME):
            continue
        existing_files.append(rel)
    _create_backup(install_dir, existing_files)

    _write_flag(install_dir, json.dumps({"mode": "full"}))

    current_exe = install_dir / exe_name
    backup_exe = install_dir / (exe_name + ".old")

    if current_exe.exists():
        try:
            if backup_exe.exists():
                backup_exe.unlink()
        except Exception:
            pass
        try:
            current_exe.rename(backup_exe)
        except Exception:
            pass

    for item in Path(source_dir).rglob("*"):
        if item.is_file():
            rel = item.relative_to(source_dir)
            dest = install_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(item, dest)
            except Exception:
                pass


# ── process handoff ────────────────────────────────────────────────────────

def launch_apply_and_exit(
    update_mode: str,
    temp_dir: str,
    pid: int,
) -> None:
    """Spawn the current exe in ``--apply-update`` mode and exit this process.

    *update_mode* is ``"delta"`` or ``"full"``.
    *temp_dir* is the directory holding either the delta files or the full zip.
    """
    exe = sys.executable
    args = [
        exe,
        "--apply-update",
        "--update-mode", update_mode,
        "--update-temp", str(Path(temp_dir).resolve()),
        "--old-pid", str(int(pid)),
    ]
    flags = 0
    if os.name == "nt":
        flags = (
            0x08000000  # CREATE_NO_WINDOW
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        )
    subprocess.Popen(args, creationflags=flags, close_fds=True)
    sys.exit(0)


# ── --apply-update entry point ─────────────────────────────────────────────

def _wait_for_pid(pid: int, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.5)


def _parse_apply_args() -> dict:
    """Extract ``--apply-update`` flags from sys.argv."""
    result: dict = {}
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--update-mode" and i + 1 < len(args):
            result["mode"] = args[i + 1]
            i += 2
        elif args[i] == "--update-temp" and i + 1 < len(args):
            result["temp"] = args[i + 1]
            i += 2
        elif args[i] == "--old-pid" and i + 1 < len(args):
            try:
                result["pid"] = int(args[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1
    return result


def apply_update_from_args() -> None:
    """Full ``--apply-update`` handler.

    1. Wait for original PID to exit.
    2. Apply delta or full update.
    3. Remove flag on success.
    4. Relaunch with ``--skip-update-once``.
    """
    parsed = _parse_apply_args()
    mode = parsed.get("mode", "")
    temp_dir = parsed.get("temp", "")
    old_pid = parsed.get("pid")

    if not mode or not temp_dir or old_pid is None:
        sys.exit(1)

    _wait_for_pid(old_pid)

    install_dir = _install_dir()
    exe_name = Path(sys.executable).name
    temp_path = Path(temp_dir)

    try:
        if mode == "delta":
            # temp_dir holds individual files mirroring install structure
            downloaded: list[tuple[str, Path]] = []
            for item in temp_path.rglob("*"):
                if item.is_file():
                    rel = str(item.relative_to(temp_path))
                    downloaded.append((rel, item))
            _apply_delta(downloaded, install_dir, exe_name)
        elif mode == "full":
            # temp_dir should contain a .pkg zip (or already-extracted folder)
            pkg_candidates = list(temp_path.glob("*.pkg"))
            if pkg_candidates:
                extract_dir = temp_path / "extracted"
                source_dir = extract_full_package(pkg_candidates[0], extract_dir)
            else:
                # Assume temp_dir IS the extracted content
                source_dir = temp_path
            _apply_full(source_dir, install_dir, exe_name)
        else:
            sys.exit(2)

        # Success – remove flag
        try:
            _flag_path(install_dir).unlink(missing_ok=True)
        except Exception:
            pass
    finally:
        # Clean up temp files
        try:
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
        except Exception:
            pass

    # Relaunch
    new_exe = install_dir / exe_name
    if new_exe.exists():
        relaunch_flags = 0
        if os.name == "nt":
            relaunch_flags = 0x08000000  # CREATE_NO_WINDOW
        subprocess.Popen(
            [str(new_exe), "--skip-update-once"],
            creationflags=relaunch_flags,
            close_fds=True,
        )
    sys.exit(0)
