from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path


APP_FOLDER_NAME = "SoundboardEZ"
STATE_FILE_NAME = "app_state.json"


@dataclass
class AppState:
    startupEnabled: bool = False
    startSoundboardOnLaunch: bool = True
    soundboardEnabled: bool = True
    allowNotifications: bool = False
    autoUpdateEnabled: bool = True
    lastUpdateCheckUtc: str = ""
    lastUpdateAttemptUtc: str = ""
    lastUpdateVersionSeen: str = ""


def _state_file_path() -> Path:
    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        root = Path(appdata)
    else:
        root = Path.home()
    folder = root / APP_FOLDER_NAME
    folder.mkdir(parents=True, exist_ok=True)
    return folder / STATE_FILE_NAME


def load_app_state() -> AppState:
    path = _state_file_path()
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        return AppState()

    if not isinstance(data, dict):
        return AppState()

    state = AppState()
    state.startupEnabled = bool(data.get("startupEnabled", state.startupEnabled))
    state.startSoundboardOnLaunch = bool(
        data.get("startSoundboardOnLaunch", state.startSoundboardOnLaunch)
    )
    state.soundboardEnabled = bool(data.get("soundboardEnabled", state.soundboardEnabled))
    state.allowNotifications = bool(data.get("allowNotifications", state.allowNotifications))
    state.autoUpdateEnabled = bool(data.get("autoUpdateEnabled", state.autoUpdateEnabled))
    state.lastUpdateCheckUtc = str(data.get("lastUpdateCheckUtc", state.lastUpdateCheckUtc) or "")
    state.lastUpdateAttemptUtc = str(data.get("lastUpdateAttemptUtc", state.lastUpdateAttemptUtc) or "")
    state.lastUpdateVersionSeen = str(data.get("lastUpdateVersionSeen", state.lastUpdateVersionSeen) or "")
    return state


def save_app_state(state: AppState) -> None:
    path = _state_file_path()
    payload = json.dumps(asdict(state), indent=2, sort_keys=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
