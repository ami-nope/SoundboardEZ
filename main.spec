# -*- mode: python ; coding: utf-8 -*-
# SoundboardEZ – PyInstaller onedir build
# Build: pyinstaller main.spec

import os
import shutil

block_cipher = None

# ── locate bundled binaries ────────────────────────────────────────────────

ffmpeg_exe = shutil.which("ffmpeg")
extra_binaries = []
if ffmpeg_exe:
    extra_binaries.append((ffmpeg_exe, "."))

# ── analysis ───────────────────────────────────────────────────────────────

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=extra_binaries,
    datas=[
        ('assets', 'assets'),
    ],
    hiddenimports=[
        'PyQt6.sip',
        'pyrnnoise',
        'engineio.async_drivers.threading',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SoundboardEZ',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/app.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SoundboardEZ',
)
