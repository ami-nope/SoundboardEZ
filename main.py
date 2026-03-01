from __future__ import annotations

import sys


def main() -> int:
    # --apply-update mode: wait for old process, replace files, relaunch, exit.
    if "--apply-update" in sys.argv:
        from updater import apply_update_from_args

        apply_update_from_args()
        return 0  # unreachable – apply_update_from_args calls sys.exit

    # Crash recovery: if a previous update was interrupted, roll back.
    if getattr(sys, "frozen", False):
        from updater import check_and_rollback_on_startup, clear_update_flag

        rolled_back = check_and_rollback_on_startup()
        if rolled_back:
            print("[main] Rolled back interrupted update.")

        # If we reach here the app launched successfully – clear the flag.
        clear_update_flag()

    # Normal startup (includes update check for frozen builds).
    from ui import run_ui

    return run_ui()


if __name__ == "__main__":
    raise SystemExit(main())
