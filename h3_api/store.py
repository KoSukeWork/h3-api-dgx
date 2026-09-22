"""SQLite is the queue and the record of public IDs; backend IDs stay internal."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class QueueFull(Exception):
    pass


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT UNIQUE NOT NULL, request TEXT NOT NULL,
                    status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                    error TEXT, output TEXT
                );
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, path TEXT NOT NULL,
                    bytes INTEGER NOT NULL, created REAL NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def recover(self):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='failed', error=?, updated=? WHERE status IN ('loading','running')",
                ("service_interrupted: submit again to retry", time.time()),
            )

    def submit(self, request: dict, limit: int) -> str:
        job_id = f"video_{uuid.uuid4().hex}"
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('queued','loading','running')"
            ).fetchone()[0]
            if count >= limit:
                raise QueueFull()
            db.execute(
                "INSERT INTO jobs(id,request,status,created,updated) VALUES(?,?,'queued',?,?)",
                (job_id, json.dumps(request), now, now),
            )
        return job_id

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY sequence LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='loading', updated=? WHERE id=?", (time.time(), row["id"]))
            return dict(row)

    def update(self, job_id, status, *, error=None, output=None):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=?,error=?,output=?,updated=? WHERE id=?",
                (status, error, output, time.time(), job_id),
            )

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def file(self, file_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()
            return dict(row) if row else None

    def add_file(self, file_id, kind, path, size):
        with self.connect() as db:
            db.execute("INSERT INTO files VALUES(?,?,?,?,?)", (file_id, kind, str(path), size, time.time()))

    def counts(self):
        with self.connect() as db:
            return dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status").fetchall())
