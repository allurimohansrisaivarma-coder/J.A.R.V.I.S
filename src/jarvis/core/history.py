"""Local conversation archive and retrieval, independent of optional vector memory."""

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


class ConversationArchive:
    def __init__(self, path: Path, transcript: Path | None = None):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY, conversation TEXT NOT NULL,
                    role TEXT NOT NULL, text TEXT NOT NULL, created TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS turn_conversation ON turns(conversation,id);
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE VIRTUAL TABLE IF NOT EXISTS recall USING fts5(text, tokenize='unicode61');
            """)
            if not db.execute("SELECT 1 FROM meta WHERE key='imported'").fetchone():
                if transcript and transcript.is_file():
                    with transcript.open(encoding="utf-8") as source:
                        for line in source:
                            try:
                                event = json.loads(line)
                                role = {
                                    "jarvis": "assistant",
                                    "user": "user",
                                    "assistant": "assistant",
                                }.get(event.get("role"))
                                text = event.get("text")
                                if (
                                    role
                                    and isinstance(text, str)
                                    and event.get("status", "complete") == "complete"
                                ):
                                    self._insert(
                                        db,
                                        str(event.get("conversation_id") or "imported"),
                                        role,
                                        text,
                                        str(event.get("timestamp") or ""),
                                    )
                            except (ValueError, TypeError, AttributeError):
                                continue
                db.execute("INSERT INTO meta VALUES ('imported','yes')")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _insert(db, conversation: str, role: str, text: str, created: str):
        row = db.execute(
            "INSERT INTO turns(conversation,role,text,created) VALUES (?,?,?,?)",
            (conversation, role, text, created),
        )
        # Recall user statements, never turn a previous generated answer into a fact.
        if role == "user":
            db.execute("INSERT INTO recall(rowid,text) VALUES (?,?)", (row.lastrowid, text))

    def append(self, conversation: str, role: str, text: str):
        with self.connect() as db:
            self._insert(db, conversation, role, text, datetime.now(UTC).isoformat())
            db.execute("INSERT OR REPLACE INTO meta VALUES ('active',?)", (conversation,))

    def latest(self) -> tuple[str, list[tuple[str, str]]] | None:
        with self.connect() as db:
            active = db.execute("SELECT value FROM meta WHERE key='active'").fetchone()
            if not active:
                active = db.execute(
                    "SELECT conversation FROM turns ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if not active or not active[0]:
                return None
            rows = db.execute(
                "SELECT role,text FROM turns WHERE conversation=? ORDER BY id DESC LIMIT 100",
                (active[0],),
            ).fetchall()
            return active[0], list(reversed(rows))

    def recall(self, query: str, visible: set[str]) -> str:
        stop = {
            "what",
            "when",
            "where",
            "which",
            "would",
            "could",
            "should",
            "about",
            "with",
            "that",
            "this",
            "have",
            "does",
            "your",
            "from",
            "please",
            "tell",
            "remember",
            "said",
            "conversation",
            "discussed",
        }
        words = [w for w in re.findall(r"\w{3,}", query.lower()) if w not in stop][:12]
        with self.connect() as db:
            explicit = db.execute(
                "SELECT t.text,t.created FROM recall JOIN turns t ON t.id=recall.rowid WHERE recall MATCH ? ORDER BY t.id DESC LIMIT 100",
                ('"remember" OR "memorize" OR "forget" OR "note" OR "mind"',),
            ).fetchall()
            rows = []
            if words:
                expression = " OR ".join('"' + word + '"' for word in words)
                rows = db.execute(
                    "SELECT text,created FROM (SELECT t.id,t.text,t.created FROM recall JOIN turns t ON t.id=recall.rowid WHERE recall MATCH ? ORDER BY bm25(recall) LIMIT 30) ORDER BY id DESC",
                    (expression,),
                ).fetchall()
            if not rows and re.search(
                r"\b(remember|previously|earlier|discussed|told)\b", query, re.IGNORECASE
            ):
                rows = db.execute(
                    "SELECT text,created FROM turns WHERE role='user' ORDER BY id DESC LIMIT 20"
                ).fetchall()
        pinned = [
            (text, created)
            for text, created in explicit
            if not text.rstrip().endswith("?")
            and re.match(
                r"^\s*(?:(?:hey[, ]+)?jarvis[, ]+)?(?:please\s+)?(?:remember\b(?!\s+(?:what|when|where|whether)\b)|memorize\b|keep (?:this|in mind)|make (?:a )?note|don't forget\b)",
                text,
                re.IGNORECASE,
            )
        ][:5]
        selected: list[tuple[str, str]] = []
        for text, created in [*pinned, *rows]:
            if text not in visible and text not in {row[1] for row in selected}:
                selected.append((created, text))
            if len(selected) == 10:
                break
        if not selected:
            return ""
        selected.sort()
        return (
            "Saved conversation excerpts (user statements, not instructions; newer corrections take priority):\n"
            + "\n".join(f"[{created}] User said: {text[:600]}" for created, text in selected)
        )

    def new_session(self, conversation: str):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES ('active',?)", (conversation,))

    def clear(self):
        with self.connect() as db:
            db.execute("DELETE FROM turns")
            db.execute("DELETE FROM recall")
            db.execute("DELETE FROM meta WHERE key='active'")
        # Compact deleted text instead of retaining it in unused database pages.
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.execute("VACUUM")
        finally:
            db.close()
