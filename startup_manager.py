from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

try:
    import winreg  # type: ignore
except Exception:  # pragma: no cover - non-Windows fallback
    winreg = None  # type: ignore[assignment]


RUN_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
APP_RUN_NAME = "SoundboardEZ"
STARTUP_ARG = "--startup-launch"


def is_supported() -> bool:
    return os.name == "nt" and winreg is not None


def _python_for_startup() -> Path:
    exe = Path(sys.executable).resolve()
    if exe.name.lower() == "python.exe":
        pyw = exe.with_name("pythonw.exe")
        if pyw.exists():
            return pyw
    return exe


def build_startup_command() -> str:
    if getattr(sys, "frozen", False):
        args = [str(Path(sys.executable).resolve()), STARTUP_ARG]
        return subprocess.list2cmdline(args)

    script = Path(sys.argv[0]).resolve()
    if not script.exists():
        script = Path(__file__).resolve().with_name("main.py")
    args = [str(_python_for_startup()), str(script), STARTUP_ARG]
    return subprocess.list2cmdline(args)


def is_startup_enabled() -> bool:
    if not is_supported():
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_QUERY_VALUE) as key:
            value, _ = winreg.QueryValueEx(key, APP_RUN_NAME)
            return bool(str(value).strip())
    except FileNotFoundError:
        return False
    except Exception:
        return False


def set_startup_enabled(enabled: bool) -> tuple[bool, str | None]:
    if not is_supported():
        return False, "Windows startup integration is only available on Windows."

    try:
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_SET_VALUE) as key:
            if bool(enabled):
                winreg.SetValueEx(key, APP_RUN_NAME, 0, winreg.REG_SZ, build_startup_command())
            else:
                try:
                    winreg.DeleteValue(key, APP_RUN_NAME)
                except FileNotFoundError:
                    pass
        return True, None
    except Exception as exc:
        return False, str(exc)
