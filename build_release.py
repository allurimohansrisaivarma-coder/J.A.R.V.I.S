"""Run the release build with a durable log and machine-readable completion status."""

import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
status = root / "logs" / "build-status.json"
status.write_text(json.dumps({"state": "building"}), encoding="utf-8")
try:
    with (root / "build-repair-final.log").open("w", encoding="utf-8") as log:
        for command in (
            [sys.executable, "-m", "PyInstaller", "--noconfirm", "JARVIS.spec"],
            [sys.executable, "build.py", "--package-only"],
        ):
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    status.write_text(json.dumps({"state": "complete", "ok": True}), encoding="utf-8")
except Exception as exc:
    status.write_text(
        json.dumps({"state": "failed", "ok": False, "error": str(exc)}), encoding="utf-8"
    )
    raise
