"""Local file context source — full desktop access."""

from pathlib import Path
import re
import structlog
from jarvis.context.base import ContextSource

logger = structlog.get_logger(__name__)

# Directories JARVIS is allowed to scan and read
USER_HOME = Path.home()
ALLOWED_ROOTS = [
    USER_HOME / "Desktop",
    USER_HOME / "Documents",
    USER_HOME / "Downloads",
    USER_HOME / "OneDrive" / "Desktop",
    USER_HOME / "OneDrive" / "Documents",
    USER_HOME / "JarvisWorkspace",
]

# File extensions we can safely read as text
TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".json", ".yaml", ".yml",
    ".csv", ".xml", ".toml", ".cfg", ".ini", ".log", ".sh", ".bat", ".ps1",
    ".java", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".rb", ".php",
    ".sql", ".env", ".gitignore", ".dockerfile",
}

# Max depth to prevent scanning massive trees
MAX_DEPTH = 4
MAX_FILES_LISTED = 40


class FileContextSource(ContextSource):
    """Reads local files from the user's Desktop, Documents, and Downloads."""

    @property
    def name(self) -> str:
        return "Local Files"

    async def can_handle(self, query: str) -> bool:
        """Trigger on file/folder/path keywords OR explicit path separators."""
        q = query.lower()
        triggers = [
            "file", "folder", "directory", "desktop", "document", "download",
            "read", "open", "list", "show me", "what's on", "what is on",
        ]
        return any(t in q for t in triggers) or "\\" in query or "/" in query

    async def gather_context(self, query: str) -> str:
        q = query.lower()

        # 1. Check if query contains an explicit absolute path
        explicit_path = self._extract_path(query)
        if explicit_path and explicit_path.exists():
            if explicit_path.is_file():
                return self._read_file(explicit_path)
            elif explicit_path.is_dir():
                return self._list_directory(explicit_path)

        # 2. Determine which roots to scan based on keywords
        roots_to_scan = []
        if "desktop" in q:
            roots_to_scan = [r for r in ALLOWED_ROOTS if "desktop" in str(r).lower()]
        elif "document" in q:
            roots_to_scan = [r for r in ALLOWED_ROOTS if "document" in str(r).lower()]
        elif "download" in q:
            roots_to_scan = [r for r in ALLOWED_ROOTS if "download" in str(r).lower()]
        else:
            roots_to_scan = ALLOWED_ROOTS

        # 3. Gather all files across the selected roots
        all_files = []
        for root in roots_to_scan:
            if root.exists():
                all_files.extend(self._walk(root, depth=0))

        if not all_files:
            searched = ", ".join(str(r) for r in roots_to_scan)
            return f"No files found in: {searched}"

        # 4. Match meaningful query terms against normalised filenames.  This
        # lets "my CV" find "mycv.pdf" without reading binary files or
        # dumping their contents into the model context.
        query_terms = {
            re.sub(r"[^a-z0-9]", "", term)
            for term in re.findall(r"[a-z0-9]+", q)
            if len(term) >= 2
            and term not in {
                "can", "you", "the", "where", "have", "file", "saved", "save", "desktop",
                "please", "what", "which", "with", "from", "that", "this", "there", "my", "is",
            }
        }
        mentioned = []
        for file_path in all_files:
            normalised_name = re.sub(r"[^a-z0-9]", "", file_path.stem.lower())
            if any(term in normalised_name for term in query_terms):
                mentioned.append(file_path)

        if mentioned:
            matches = "\n".join(self._describe_file(file_path) for file_path in mentioned[:5])
            return f"Matching local files:\n{matches}"

        # 5. Fallback: list available files
        lines = []
        for f in all_files[:MAX_FILES_LISTED]:
            # Show path relative to home for readability
            try:
                rel = f.relative_to(USER_HOME)
            except ValueError:
                rel = f
            lines.append(f"  - ~/{rel}")

        header = f"Files available on the user's system ({len(all_files)} total, showing first {min(len(all_files), MAX_FILES_LISTED)}):"
        return header + "\n" + "\n".join(lines)

    # ── Helpers ──

    def _extract_path(self, query: str) -> Path | None:
        """Try to find an absolute path in the query string."""
        import re
        # Match Windows paths like C:\... or Unix paths like /home/...
        match = re.search(r'([A-Za-z]:\\[^\s"\']+|/[^\s"\']+)', query)
        if match:
            return Path(match.group(1))
        return None

    def _walk(self, root: Path, depth: int) -> list[Path]:
        """Recursively list files up to MAX_DEPTH."""
        if depth > MAX_DEPTH:
            return []
        files = []
        try:
            for entry in root.iterdir():
                # Skip hidden files/dirs
                if entry.name.startswith("."):
                    continue
                if entry.is_file():
                    files.append(entry)
                elif entry.is_dir() and not entry.name.startswith("__"):
                    files.extend(self._walk(entry, depth + 1))
        except PermissionError:
            pass
        except Exception as e:
            logger.debug("Error scanning directory", path=str(root), error=str(e))
        return files

    def _read_file(self, path: Path) -> str:
        """Read a single file, returning its content or a description."""
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            size_kb = path.stat().st_size / 1024
            return f"File: {path.name} ({path.suffix}, {size_kb:.1f} KB) — binary file, cannot display contents."

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 15000:
                content = content[:15000] + "\n...[TRUNCATED]"
            return f"File: {path}\n```\n{content}\n```"
        except Exception as e:
            return f"File: {path}\n[Error reading: {e}]"

    @staticmethod
    def _describe_file(path: Path) -> str:
        """Return safe metadata for a found file, never its raw content."""
        try:
            size_kb = path.stat().st_size / 1024
            return f"- {path} ({path.suffix.upper().lstrip('.') or 'file'}, {size_kb:.1f} KB)"
        except OSError:
            return f"- {path}"

    def _list_directory(self, directory: Path) -> str:
        """List the contents of a specific directory."""
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            lines = []
            for entry in entries[:MAX_FILES_LISTED]:
                prefix = "📁" if entry.is_dir() else "📄"
                size = ""
                if entry.is_file():
                    size_kb = entry.stat().st_size / 1024
                    size = f" ({size_kb:.1f} KB)"
                lines.append(f"  {prefix} {entry.name}{size}")
            header = f"Contents of {directory} ({len(entries)} items):"
            return header + "\n" + "\n".join(lines)
        except Exception as e:
            return f"Error listing {directory}: {e}"
