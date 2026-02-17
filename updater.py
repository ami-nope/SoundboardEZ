from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import uuid

import requests


GITHUB_OWNER = "ami-nope"
GITHUB_REPO = "SoundboardEZ"
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
INSTALLER_ASSET_NAME = "SoundboardEZ-Setup.exe"
CHECKSUM_ASSET_NAME = "SoundboardEZ-Setup.exe.sha256"
HTTP_TIMEOUT = (8.0, 30.0)


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    tag: str
    installer_url: str
    checksum_url: str


def _normalize_version(value: str) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("v"):
        text = text[1:]
    return text


def _version_tuple(value: str) -> tuple[int, ...]:
    normalized = _normalize_version(value)
    parts = re.findall(r"\d+", normalized)
    if not parts:
        return (0,)
    return tuple(int(part) for part in parts)


def compare_versions(left: str, right: str) -> int:
    a = _version_tuple(left)
    b = _version_tuple(right)
    width = max(len(a), len(b))
    a = a + (0,) * (width - len(a))
    b = b + (0,) * (width - len(b))
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


def _asset_download_url(release_data: dict, asset_name: str) -> str:
    assets = release_data.get("assets")
    if not isinstance(assets, list):
        return ""
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        if str(asset.get("name", "")).strip() != asset_name:
            continue
        return str(asset.get("browser_download_url", "")).strip()
    return ""


def check_for_update(current_version: str) -> UpdateInfo | None:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "SoundboardEZ-Updater",
    }
    response = requests.get(LATEST_RELEASE_API, headers=headers, timeout=HTTP_TIMEOUT)
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid release payload from update server.")

    tag = str(payload.get("tag_name", "")).strip()
    if not tag:
        raise RuntimeError("Release payload missing tag_name.")

    release_version = _normalize_version(tag)
    if compare_versions(release_version, current_version) <= 0:
        return None

    installer_url = _asset_download_url(payload, INSTALLER_ASSET_NAME)
    checksum_url = _asset_download_url(payload, CHECKSUM_ASSET_NAME)
    if not installer_url or not checksum_url:
        raise RuntimeError("Latest release is missing required installer/checksum assets.")

    return UpdateInfo(
        version=release_version,
        tag=tag,
        installer_url=installer_url,
        checksum_url=checksum_url,
    )


def _parse_checksum(checksum_text: str) -> str:
    match = re.search(r"\b([0-9a-fA-F]{64})\b", str(checksum_text))
    if match is None:
        raise ValueError("Checksum file does not contain a SHA256 hash.")
    return match.group(1).lower()


def verify_sha256(file_path: Path, checksum_text: str) -> bool:
    expected = _parse_checksum(checksum_text)
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    actual = digest.hexdigest().lower()
    return actual == expected


def download_update(
    update: UpdateInfo,
    dest_dir: Path,
    progress_cb=None,
) -> Path:
    target_dir = Path(dest_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    installer_path = target_dir / f"SoundboardEZ-Setup-{update.version}.exe"
    partial_path = installer_path.with_suffix(".exe.part")

    for path in (installer_path, partial_path):
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass

    checksum_response = requests.get(update.checksum_url, timeout=HTTP_TIMEOUT)
    checksum_response.raise_for_status()
    checksum_text = checksum_response.text
    _parse_checksum(checksum_text)

    downloaded = 0
    total_size: int | None = None
    try:
        with requests.get(update.installer_url, stream=True, timeout=HTTP_TIMEOUT) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length", "").strip()
            if content_length.isdigit():
                total_size = int(content_length)
            if callable(progress_cb):
                progress_cb(downloaded, total_size)

            with partial_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if callable(progress_cb):
                        progress_cb(downloaded, total_size)
        partial_path.replace(installer_path)
    except Exception:
        try:
            partial_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    if not verify_sha256(installer_path, checksum_text):
        try:
            installer_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError("Downloaded installer failed checksum verification.")

    if callable(progress_cb):
        progress_cb(total_size or downloaded, total_size or downloaded)
    return installer_path


def _ps_quote(value: str) -> str:
    return str(value).replace("'", "''")


def schedule_installer_handoff(installer_path: Path, install_dir: Path, old_pid: int) -> bool:
    installer = Path(installer_path).resolve()
    install_root = Path(install_dir).resolve()
    if not installer.exists():
        return False

    script_path = Path(tempfile.gettempdir()) / f"SoundboardEZ_update_{uuid.uuid4().hex}.ps1"
    script_body = textwrap.dedent(
        f"""
        $ErrorActionPreference = 'SilentlyContinue'
        $pidToWait = {int(old_pid)}
        $installer = '{_ps_quote(str(installer))}'
        $installDir = '{_ps_quote(str(install_root))}'

        for ($i = 0; $i -lt 120; $i++) {{
            $proc = Get-Process -Id $pidToWait -ErrorAction SilentlyContinue
            if ($null -eq $proc) {{
                break
            }}
            Start-Sleep -Milliseconds 500
        }}

        if (-not (Test-Path -LiteralPath $installer)) {{
            exit 2
        }}

        $args = @('/S', '/UPDATE=1', '/SKIPVBCABLE=1', '/AUTOLAUNCH=1', ('/D=' + $installDir))
        $proc = Start-Process -FilePath $installer -ArgumentList $args -Wait -PassThru
        $exitCode = 0
        if ($null -ne $proc) {{
            $exitCode = [int]$proc.ExitCode
        }}

        Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 500
        Remove-Item -LiteralPath $MyInvocation.MyCommand.Path -Force -ErrorAction SilentlyContinue
        exit $exitCode
        """
    ).strip() + "\n"

    try:
        script_path.write_text(script_body, encoding="utf-8")
    except Exception:
        return False

    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            )
        subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-WindowStyle",
                "Hidden",
                "-File",
                str(script_path),
            ],
            creationflags=creationflags,
            close_fds=True,
        )
        return True
    except Exception:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False
