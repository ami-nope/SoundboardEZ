from __future__ import annotations

from dataclasses import dataclass

import requests


MANIFEST_URL = "https://soundboardez.up.railway.app/manifest"
HTTP_TIMEOUT = (8.0, 30.0)


@dataclass(frozen=True)
class DeltaFileEntry:
    """One file to patch during a delta update."""
    relative_path: str
    url: str


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    mandatory: bool
    is_delta: bool
    # Full-update fields (used when is_delta is False)
    full_url: str = ""
    # Delta-update fields (used when is_delta is True)
    delta_files: tuple[DeltaFileEntry, ...] = ()


def _parse_version(value: str) -> list[int]:
    """Split a version string like '1.2.3' into a list of ints."""
    text = str(value or "").strip()
    if text.lower().startswith("v"):
        text = text[1:]
    if not text:
        return [0]

    parts: list[int] = []
    for token in text.split("."):
        piece = token.strip()
        if not piece:
            parts.append(0)
            continue
        try:
            parts.append(int(piece))
        except ValueError:
            digits = "".join(ch for ch in piece if ch.isdigit())
            parts.append(int(digits) if digits else 0)
    return parts or [0]


def is_server_version_newer(server_version: str, current_version: str) -> bool:
    """Semantic comparison: True only if *server_version* is strictly newer."""
    server_parts = _parse_version(server_version)
    current_parts = _parse_version(current_version)

    width = max(len(server_parts), len(current_parts))
    server_parts.extend([0] * (width - len(server_parts)))
    current_parts.extend([0] * (width - len(current_parts)))
    return server_parts > current_parts


def _versions_equal(a: str, b: str) -> bool:
    """True when both version strings represent the same semantic version."""
    pa = _parse_version(a)
    pb = _parse_version(b)
    width = max(len(pa), len(pb))
    pa.extend([0] * (width - len(pa)))
    pb.extend([0] * (width - len(pb)))
    return pa == pb


def check_for_update(current_version: str) -> UpdateInfo | None:
    """Fetch the remote manifest and return *UpdateInfo* when an update is available.

    Decision logic
    ──────────────
    • If ``manifest["delta"]["from"]`` matches *current_version* → delta update.
    • Otherwise → full update (using ``manifest["full"]["url"]``).
    """
    response = requests.get(MANIFEST_URL, timeout=HTTP_TIMEOUT)

    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError("Manifest response is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Invalid manifest payload from update server.")

    if response.status_code >= 400:
        error = str(payload.get("error", "")).strip()
        raise RuntimeError(error or f"Manifest request failed (HTTP {response.status_code}).")

    latest_version = str(payload.get("version", "")).strip().lstrip("vV")
    if not latest_version:
        raise RuntimeError("Manifest payload missing version.")

    if not is_server_version_newer(latest_version, current_version):
        return None

    # --- mandatory flag ---
    mandatory = False
    raw_mandatory = payload.get("mandatory", False)
    if isinstance(raw_mandatory, bool):
        mandatory = raw_mandatory
    elif isinstance(raw_mandatory, (int, float)):
        mandatory = raw_mandatory != 0
    else:
        mandatory = str(raw_mandatory).strip().lower() in {"1", "true", "yes", "on"}

    # --- delta section ---
    delta_section = payload.get("delta")
    can_delta = False
    delta_files: list[DeltaFileEntry] = []

    if isinstance(delta_section, dict):
        delta_from = str(delta_section.get("from", "")).strip().lstrip("vV")
        if delta_from and _versions_equal(delta_from, current_version):
            files_map = delta_section.get("files")
            if isinstance(files_map, dict) and files_map:
                for rel_path, entry in files_map.items():
                    if not isinstance(entry, dict):
                        continue
                    url = str(entry.get("url", "")).strip()
                    if url:
                        delta_files.append(DeltaFileEntry(relative_path=str(rel_path), url=url))
                if delta_files:
                    can_delta = True

    if can_delta:
        return UpdateInfo(
            version=latest_version,
            mandatory=mandatory,
            is_delta=True,
            delta_files=tuple(delta_files),
        )

    # --- full section ---
    full_section = payload.get("full")
    if not isinstance(full_section, dict):
        raise RuntimeError("Manifest missing 'full' section and delta is not applicable.")

    full_url = str(full_section.get("url", "")).strip()
    if not full_url:
        raise RuntimeError("Manifest 'full' section missing url.")

    return UpdateInfo(
        version=latest_version,
        mandatory=mandatory,
        is_delta=False,
        full_url=full_url,
    )

