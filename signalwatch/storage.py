"""Small local SQLite store; uploaded CSV content never leaves this directory."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, runtime: Path):
        self.runtime = Path(runtime).resolve()
        self.upload_dir = self.runtime / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.database = self.runtime / "sessions.sqlite3"
        with self.connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, raw_file TEXT NOT NULL,
                    mapping TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, cursor INTEGER NOT NULL,
                    total INTEGER NOT NULL, model_sha TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, message TEXT NOT NULL,
                    created_at TEXT NOT NULL, finished_at TEXT, error TEXT
                );
                """
            )

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add_upload(self, upload_id: str, name: str, raw: bytes, mapping: dict):
        filename = uuid.uuid4().hex + ".csv"
        target = self.upload_dir / filename
        target.write_bytes(raw)
        try:
            with self.connection() as connection:
                connection.execute(
                    "INSERT INTO uploads VALUES (?, ?, ?, ?, ?)",
                    (upload_id, name, filename, json.dumps(mapping), now()),
                )
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def list_uploads(self) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM uploads ORDER BY created_at DESC").fetchall()
        return [self._upload(row) for row in rows]

    @staticmethod
    def _upload(row) -> dict:
        result = dict(row)
        result["mapping"] = json.loads(result["mapping"])
        return result

    def get_upload(self, upload_id: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone()
        return self._upload(row) if row else None

    def _upload_path(self, filename: str) -> Path:
        target = (self.upload_dir / filename).resolve()
        if target.parent != self.upload_dir or target.suffix != ".csv":
            raise ValueError("Invalid local upload reference")
        return target

    def upload_bytes(self, upload_id: str) -> bytes:
        entry = self.get_upload(upload_id)
        if entry is None:
            raise KeyError(upload_id)
        return self._upload_path(entry["raw_file"]).read_bytes()

    def delete_upload(self, upload_id: str) -> bool:
        entry = self.get_upload(upload_id)
        if entry is None:
            return False
        self._upload_path(entry["raw_file"]).unlink(missing_ok=True)
        with self.connection() as connection:
            connection.execute("DELETE FROM sessions WHERE dataset_id=?", (upload_id,))
            connection.execute("DELETE FROM uploads WHERE id=?", (upload_id,))
        return True

    def create_session(self, dataset_id: str, total: int, model_sha: str) -> dict:
        session_id = uuid.uuid4().hex
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, dataset_id, 0, total, model_sha, now()),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> list[dict]:
        with self.connection() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM sessions ORDER BY created_at DESC")]

    def update_cursor(self, session_id: str, cursor: int):
        with self.connection() as connection:
            connection.execute(
                "UPDATE sessions SET cursor=MIN(total, MAX(0, ?)) WHERE id=?", (cursor, session_id)
            )

    def advance_cursor(self, session_id: str, count: int) -> dict | None:
        with self.connection() as connection:
            connection.execute(
                "UPDATE sessions SET cursor=MIN(total, cursor+?) WHERE id=?", (count, session_id)
            )
        return self.get_session(session_id)

    def reset_session(self, session_id: str, model_sha: str):
        with self.connection() as connection:
            connection.execute(
                "UPDATE sessions SET cursor=0, model_sha=? WHERE id=?", (model_sha, session_id)
            )

    def delete_session(self, session_id: str) -> bool:
        with self.connection() as connection:
            return connection.execute("DELETE FROM sessions WHERE id=?", (session_id,)).rowcount > 0

    def create_job(self) -> dict:
        job_id = uuid.uuid4().hex
        with self.connection() as connection:
            connection.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, NULL, NULL)", (job_id, "queued", "Queued", now()))
        return self.get_job(job_id)

    def update_job(self, job_id: str, status: str, message: str, error: str | None = None):
        finished = now() if status in {"completed", "failed"} else None
        with self.connection() as connection:
            connection.execute(
                "UPDATE jobs SET status=?, message=?, finished_at=?, error=? WHERE id=?",
                (status, message, finished, error, job_id),
            )

    def get_job(self, job_id: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def recover_jobs(self):
        with self.connection() as connection:
            connection.execute(
                "UPDATE jobs SET status='failed', message=?, finished_at=? WHERE status IN ('running','queued')",
                ("Server restarted before training finished. Run training again.", now()),
            )
