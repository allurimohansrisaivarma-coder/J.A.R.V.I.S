"""Safe local-file context for user-approved directories."""

import re
from pathlib import Path

import structlog

from jarvis.context.base import ContextSource

logger = structlog.get_logger(__name__)

USER_HOME = Path.home()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_ROOTS = [
    USER_HOME / "Desktop",
    USER_HOME / "Documents",
    USER_HOME / "Downloads",
    USER_HOME / "OneDrive" / "Desktop",
    USER_HOME / "OneDrive" / "Documents",
    USER_HOME / "JarvisWorkspace",
    PROJECT_ROOT,
]

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".xml",
    ".toml",
    ".cfg",
    ".ini",
    ".log",
    ".sh",
    ".bat",
    ".ps1",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".go",
    ".rb",
    ".php",
    ".sql",
    ".gitignore",
    ".dockerfile",
}
SENSITIVE_FILENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "token.json",
    "client_secret.json",
    "secrets.yaml",
    "secrets.yml",
}
MAX_DEPTH = 4
MAX_FILES_LISTED = 40
MAX_FILES_SCANNED = 1000
MAX_TEXT_CHARS = 15_000


class FileContextSource(ContextSource):
    """Reads safe text and metadata only from approved local roots."""

    @property
    def name(self) -> str:
        return "Local Files"

    async def can_handle(self, query: str) -> bool:
        """Trigger only for an explicit path or a clear local-file request."""
        q = query.lower()
        triggers = (
            "file",
            "folder",
            "directory",
            "desktop",
            "documents",
            "downloads",
            "readme",
            "local path",
            "project files",
            "repo files",
            "source code",
        )
        return any(term in q for term in triggers) or self._extract_path(query) is not None

    async def gather_context(self, query: str, **kwargs: object) -> str:
        q = query.lower()
        explicit_path = self._extract_path(query)
        if explicit_path is not None:
            if not self._is_allowed(explicit_path):
                return "That path is outside the local directories JARVIS is allowed to access."
            if not explicit_path.exists():
                return f"Local path not found: {explicit_path}"
            if explicit_path.is_file():
                return self._read_file(explicit_path)
            if explicit_path.is_dir():
                return self._list_directory(explicit_path)

        if "desktop" in q:
            roots_to_scan = [root for root in ALLOWED_ROOTS if "desktop" in str(root).lower()]
        elif "document" in q:
            roots_to_scan = [root for root in ALLOWED_ROOTS if "document" in str(root).lower()]
        elif "download" in q:
            roots_to_scan = [root for root in ALLOWED_ROOTS if "download" in str(root).lower()]
        else:
            roots_to_scan = ALLOWED_ROOTS

        all_files: list[Path] = []
        for root in roots_to_scan:
            if root.exists() and self._is_allowed(root):
                remaining = MAX_FILES_SCANNED - len(all_files)
                all_files.extend(self._walk(root, depth=0, remaining=remaining))
            if len(all_files) >= MAX_FILES_SCANNED:
                break

        if not all_files:
            return "No matching files were found in the approved local directories."

        ignored_terms = {
            "can",
            "you",
            "the",
            "where",
            "have",
            "file",
            "saved",
            "save",
            "desktop",
            "please",
            "what",
            "which",
            "with",
            "from",
            "that",
            "this",
            "there",
            "my",
            "is",
            "read",
            "open",
            "show",
            "me",
            "code",
            "project",
            "folder",
            "documents",
            "downloads",
        }
        query_terms = {
            re.sub(r"[^a-z0-9]", "", term)
            for term in re.findall(r"[a-z0-9]+", q)
            if len(term) >= 2 and term not in ignored_terms
        }
        mentioned = [
            path
            for path in all_files
            if any(term in re.sub(r"[^a-z0-9]", "", path.stem.lower()) for term in query_terms)
        ]

        if "readme" in q and PROJECT_ROOT / "README.md" not in mentioned:
            mentioned.append(PROJECT_ROOT / "README.md")

        if mentioned:
            matches = mentioned[:5]
            wants_contents = any(
                term in q for term in ("read ", "contents", "what does", "show me")
            )
            if wants_contents and len(matches) == 1 and self._is_readable_text(matches[0]):
                return self._read_file(matches[0])
            return "Matching local files:\n" + "\n".join(
                self._describe_file(path) for path in matches
            )

        lines = []
        for path in all_files[:MAX_FILES_LISTED]:
            try:
                display = Path("~") / path.relative_to(USER_HOME)
            except ValueError:
                display = path
            lines.append(f"  - {display}")
        return (
            f"Files available in approved directories ({len(all_files)} found, showing "
            f"{len(lines)}):\n" + "\n".join(lines)
        )

    def _extract_path(self, query: str) -> Path | None:
        """Extract a quoted or unquoted absolute Windows/Unix path."""
        quoted = re.search(r"[\"']([A-Za-z]:[\\/][^\"']+|/[^\"']+)[\"']", query)
        if quoted:
            return Path(quoted.group(1))
        match = re.search(r"([A-Za-z]:[\\/][^\r\n,;]+|/(?:[^\s\"']+/?)+)", query)
        if match:
            return Path(match.group(1).strip().rstrip(".?!"))
        return None

    @staticmethod
    def _resolved(path: Path) -> Path:
        return path.expanduser().resolve(strict=False)

    def _is_allowed(self, path: Path) -> bool:
        resolved = self._resolved(path)
        return any(resolved.is_relative_to(self._resolved(root)) for root in ALLOWED_ROOTS)

    @staticmethod
    def _is_sensitive(path: Path) -> bool:
        name = path.name.lower()
        return name in SENSITIVE_FILENAMES or name.startswith(".env.") or "credential" in name

    def _is_readable_text(self, path: Path) -> bool:
        return (
            self._is_allowed(path)
            and not self._is_sensitive(path)
            and path.suffix.lower() in TEXT_EXTENSIONS
        )

    def _walk(self, root: Path, depth: int, remaining: int) -> list[Path]:
        """Recursively list a bounded number of files without crossing allowed roots."""
        if depth > MAX_DEPTH or remaining <= 0 or not self._is_allowed(root):
            return []
        files: list[Path] = []
        try:
            for entry in root.iterdir():
                if len(files) >= remaining:
                    break
                if entry.name.startswith(".") or not self._is_allowed(entry):
                    continue
                if entry.is_file() and not self._is_sensitive(entry):
                    files.append(entry)
                elif entry.is_dir() and not entry.name.startswith("__"):
                    files.extend(self._walk(entry, depth + 1, remaining - len(files)))
        except PermissionError:
            logger.debug("Directory access denied", path=str(root))
        except OSError as exc:
            logger.debug("Error scanning directory", path=str(root), error=str(exc))
        return files

    def _read_file(self, path: Path) -> str:
        if not self._is_allowed(path):
            return "That path is outside the local directories JARVIS is allowed to access."
        if self._is_sensitive(path):
            return f"File: {path.name} — contents are protected because this may contain secrets."
        if not self._is_readable_text(path):
            try:
                size_kb = path.stat().st_size / 1024
            except OSError:
                size_kb = 0
            return (
                f"File: {path.name} ({path.suffix or 'unknown'}, {size_kb:.1f} KB) — "
                "binary or unsupported text format."
            )
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if len(content) > MAX_TEXT_CHARS:
                content = content[:MAX_TEXT_CHARS] + "\n...[TRUNCATED]"
            return f"File: {path}\n```\n{content}\n```"
        except OSError as exc:
            logger.warning("Failed to read local file", path=str(path), error=str(exc))
            return f"File: {path}\n[Unable to read file]"

    @staticmethod
    def _describe_file(path: Path) -> str:
        try:
            size_kb = path.stat().st_size / 1024
            return f"- {path} ({path.suffix.upper().lstrip('.') or 'file'}, {size_kb:.1f} KB)"
        except OSError:
            return f"- {path}"

    def _list_directory(self, directory: Path) -> str:
        if not self._is_allowed(directory):
            return "That path is outside the local directories JARVIS is allowed to access."
        try:
            entries = [
                entry
                for entry in directory.iterdir()
                if not entry.name.startswith(".")
                and not self._is_sensitive(entry)
                and self._is_allowed(entry)
            ]
            entries.sort(key=lambda path: (not path.is_dir(), path.name.lower()))
            lines = []
            for entry in entries[:MAX_FILES_LISTED]:
                size = f" ({entry.stat().st_size / 1024:.1f} KB)" if entry.is_file() else ""
                kind = "[DIR]" if entry.is_dir() else "[FILE]"
                lines.append(f"  {kind} {entry.name}{size}")
            return f"Contents of {directory} ({len(entries)} items):\n" + "\n".join(lines)
        except OSError as exc:
            logger.warning("Failed to list local directory", path=str(directory), error=str(exc))
            return f"Unable to list {directory}."
