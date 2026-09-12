"""Build a GitHub Release-ready Windows distribution of JARVIS."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from jarvis import __version__

ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
RELEASE_DIR = DIST_DIR / f"JARVIS-{__version__}"
ARCHIVE = DIST_DIR / f"JARVIS-Windows-x64-v{__version__}.zip"
PRIVATE_FILENAMES = {".env", "credentials.json", "token.json", "provider-keys.dat"}
RUNTIME_DIRS = {"data", "logs"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_release_docs() -> None:
    for filename in ("START-HERE.txt", "README.md", "CHANGELOG.md", "LICENSE", ".env.example"):
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
            relative = path.relative_to(RELEASE_DIR)
            is_private = (
                path.name.lower() in PRIVATE_FILENAMES
                or path.name.lower().endswith("_token.json")
                or relative.as_posix().lower() == "config/jarvis.yaml"
            )
            is_runtime_data = bool(relative.parts) and relative.parts[0].lower() in RUNTIME_DIRS
            if path.is_file() and not is_private and not is_runtime_data:
                bundle.write(path, Path("JARVIS") / relative)


def _write_checksums(executable: Path) -> Path:
    checksums = {
        executable.relative_to(DIST_DIR).as_posix(): _sha256(executable),
        ARCHIVE.name: _sha256(ARCHIVE),
    }
    checksum_file = DIST_DIR / "SHA256SUMS.txt"
    checksum_file.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )
    return checksum_file


def main(*, package_only: bool = False) -> None:
    if package_only:
        print(f"Packaging the existing JARVIS v{__version__} build...", flush=True)
    else:
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
    checksum_file = _write_checksums(executable)

    print(f"Application folder: {RELEASE_DIR}")
    print(f"GitHub Release ZIP: {ARCHIVE}")
    print(f"Checksums: {checksum_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-only",
        action="store_true",
        help="Recreate the ZIP and checksums from an existing dist/JARVIS build.",
    )
    main(package_only=parser.parse_args().package_only)
