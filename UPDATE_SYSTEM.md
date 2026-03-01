# SoundboardEZ Update System

Manifest-driven delta patching with rollback protection and crash recovery.

## Build Mode

PyInstaller **onedir** build.  The install directory is a folder tree:

```
dist/SoundboardEZ/
    SoundboardEZ.exe
    <runtime files>
```

## Manifest

Endpoint: `https://soundboardez.up.railway.app/manifest`

```json
{
  "version": "1.2.0",
  "mandatory": false,
  "full": {
    "url": "https://…/SoundboardEZ_full_1.2.0.zip"
  },
  "delta": {
    "from": "1.1.0",
    "files": {
      "SoundboardEZ.exe": { "url": "https://…/SoundboardEZ.exe" },
      "core/audio_engine.dll": { "url": "https://…/audio_engine.dll" }
    }
  }
}
```

## Version Comparison

Semantic: split by `.`, convert to int, pad with zeroes, compare numerically.

## Update Decision

1. If `current_version == manifest.delta.from` → **delta** update.
2. Otherwise → **full** update.

## Delta Patching

For each file in `delta.files`:

1. Download to temp directory.
2. Back up the original into `INSTALL_DIR/_backup/<relative_path>`.
3. Replace original file.

The install directory is **not** wiped.

## Full Update

1. Download full ZIP.
2. Extract to temp directory.
3. Back up all existing files into `_backup/`.
4. Replace entire install directory.
5. User config (`%APPDATA%/SoundboardEZ`) is preserved automatically.

## Safe Windows Handoff

1. Main process launches itself with
   `--apply-update --update-mode <delta|full> --update-temp <dir> --old-pid <pid>`.
2. Main process exits.
3. The `--apply-update` instance:
   - Waits for the original PID to fully exit.
   - Creates `update_in_progress.flag`.
   - Applies delta or full replacement (renaming running EXE → `*.old`).
   - Removes flag on success.
   - Relaunches with `--skip-update-once`.
   - Exits.

## Rollback / Crash Recovery

- Before applying, an `update_in_progress.flag` is written.
- On next startup, if the flag exists the app restores every file from `_backup/`.
- After a successful launch, the flag and `_backup/` are deleted.

## Error Handling

- HTTP, JSON, and file-permission errors are caught.
- Errors are displayed inside the UpdateDialog.
- Mandatory updates block bypass; only retry is available.

## Notes

- No GitHub Releases API dependency at runtime.
- No legacy update code remains.
