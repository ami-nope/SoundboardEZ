# -*- mode: python ; coding: utf-8 -*-

"""
PyInstaller spec for SoundboardEZ
- onefile, windowed
- bundles ffmpeg (via imageio_ffmpeg)
- bundles VB-Cable driver zip and optional assets
"""

from pathlib import Path

import imageio_ffmpeg

project_root = Path(SPECPATH)
assets_dir = project_root / "assets"
sounds_dir = project_root / "sounds"

datas = []
binaries = []
hiddenimports = ["PyQt6.sip", "pyrnnoise"]

# Icon
icon_path = assets_dir / "app.ico"
if icon_path.exists():
    icon_file = str(icon_path)
else:
    icon_file = None

# ffmpeg binary
try:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    binaries.append((ffmpeg_exe, "ffmpeg.exe"))
except Exception:
    pass

# VB-Cable driver pack
vb_zip = assets_dir / "VBCABLE_Driver_Pack43.zip"
if vb_zip.exists():
    datas.append((str(vb_zip), "VBCABLE_Driver_Pack43.zip"))

# Optional sounds folder
if sounds_dir.exists():
    for p in sounds_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(project_root)
            datas.append((str(p), str(rel)))

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="SoundboardEZ",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)
