SoundboardEZ — Offline Installer Notes
=====================================

What’s included:
- SoundboardEZ app (onefile exe)
- ffmpeg (bundled for previews)
- VB-Cable driver pack (if provided in assets)
- Optional sounds folder (if present at build time)

Install:
- Run SoundboardEZ-Setup.exe (per-user, no admin required; VB-Cable installer may prompt).
- Shortcuts are created in Start Menu and on Desktop.

VB-Cable:
- The installer unzips vb-cable files to the app folder and launches the setup.
- Accept the VB-Audio license, install the driver, and reboot if prompted.
- The app does not uninstall the VB-Cable driver; uninstall it manually if needed.

Uninstall:
- Use “SoundboardEZ” in Apps & Features or run Uninstall.exe in the install folder.
- Removes app files and shortcuts; leaves VB-Cable driver untouched.

Offline use:
- No internet is required during install or runtime.

Minimum requirements:
- Windows 10/11, per-user install under %LOCALAPPDATA%.
- Virtual audio cable (VB-Cable) for routing if you need virtual mic output.
