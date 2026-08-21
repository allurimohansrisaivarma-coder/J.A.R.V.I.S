"""Build a GitHub Release-ready Windows distribution of JARVIS."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from jarvis import __version__

ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
RELEASE_DIR = DIST_DIR / "JARVIS"
ARCHIVE = DIST_DIR / f"JARVIS-Windows-x64-v{__version__}.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_release_docs() -> None:
    for filename in ("README.md", "CHANGELOG.md", "LICENSE", ".env.example"):
        shutil.copy2(ROOT / filename, RELEASE_DIR / filename)

    config_dir = RELEASE_DIR / "config"
    config_dir.mkdir(exist_ok=True)
    shutil.copy2(
        ROOT / "src" / "jarvis" / "config" / "defaults.yaml",
        config_dir / "jarvis.example.yaml",
    )


def _create_archive() -> None:
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in RELEASE_DIR.rglob("*"):
            if path.is_file():
                bundle.write(path, Path("JARVIS") / path.relative_to(RELEASE_DIR))


def main() -> None:
    print(f"Building JARVIS v{__version__} for Windows...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "JARVIS.spec"],
        cwd=ROOT,
        check=True,
    )

    executable = RELEASE_DIR / "JARVIS.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"PyInstaller did not create {executable}")

    _copy_release_docs()
    _create_archive()

    checksums = {
        executable.relative_to(DIST_DIR).as_posix(): _sha256(executable),
        ARCHIVE.name: _sha256(ARCHIVE),
    }
    checksum_file = DIST_DIR / "SHA256SUMS.txt"
    checksum_file.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )

    print(f"Application folder: {RELEASE_DIR}")
    print(f"GitHub Release ZIP: {ARCHIVE}")
    print(f"Checksums: {checksum_file}")


if __name__ == "__main__":
    main()
