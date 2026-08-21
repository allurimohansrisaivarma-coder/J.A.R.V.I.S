"""Database connection managers for SQLite and LanceDB."""

from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

logger = structlog.get_logger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path.home() / ".jarvis" / "memory"


class DatabaseManager:
    """Manages connections to the underlying databases."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize connections to SQLite and LanceDB."""
        import lancedb

        from jarvis.memory.schema import Base, SemanticMemorySchema

        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # SQLite Setup
        sqlite_path = self.data_dir / "jarvis.db"
        self.engine = create_engine(f"sqlite:///{sqlite_path}")
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        logger.info("SQLite initialized", path=str(sqlite_path))

        # LanceDB Setup
        lancedb_path = self.data_dir / "lancedb"
        self.db = lancedb.connect(str(lancedb_path))

        # Ensure semantic memory table exists
        self.table_name = "semantic_memory"
        if self.table_name not in self.db.list_tables().tables:
            self.table = self.db.create_table(self.table_name, schema=SemanticMemorySchema)
            logger.info("LanceDB table created", name=self.table_name)
        else:
            self.table = self.db.open_table(self.table_name)
            logger.info("LanceDB initialized", path=str(lancedb_path))

    def get_sqlite_session(self) -> Session:
        """Create a caller-owned SQLAlchemy session."""
        return self.SessionLocal()

    def close(self) -> None:
        """Release database resources owned by this manager."""
        self.engine.dispose()


# Global instances
_db_manager: DatabaseManager | None = None


def init_db(data_dir: Path | None = None) -> DatabaseManager:
    """Initialize the global database manager."""
    global _db_manager
    if _db_manager is not None:
        _db_manager.close()
    _db_manager = DatabaseManager(data_dir)
    return _db_manager


def get_sqlite_session() -> Session:
    """Get a SQLite session from the global manager."""
    if not _db_manager:
        init_db()
    assert _db_manager is not None
    return _db_manager.get_sqlite_session()


def get_lancedb_table() -> Any:
    """Get the LanceDB semantic memory table."""
    if not _db_manager:
        init_db()
    assert _db_manager is not None
    return _db_manager.table
