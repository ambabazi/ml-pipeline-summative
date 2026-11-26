"""Store labelled upload bytes and metadata in SQLite."""
import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def upload_database(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(directory / "uploads.sqlite3")
    try:
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS uploads (
                    digest TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content BLOB NOT NULL
                )
            """)
            yield connection
    finally:
        connection.close()


def save_uploads(directory, label, images):
    with upload_database(directory) as connection:
        saved = 0
        duplicates = []
        for filename, content in images:
            digest = hashlib.sha256(content).hexdigest()
            cursor = connection.execute(
                "INSERT OR IGNORE INTO uploads VALUES (?, ?, ?, ?)",
                (digest, label, filename, content),
            )
            saved += cursor.rowcount
            if not cursor.rowcount:
                duplicates.append(filename)
        counts = dict(connection.execute("SELECT label, COUNT(*) FROM uploads GROUP BY label"))
    return saved, duplicates, counts


def read_uploads(directory):
    if not (Path(directory) / "uploads.sqlite3").is_file():
        return []
    with upload_database(directory) as connection:
        return connection.execute(
            "SELECT label, filename, content FROM uploads ORDER BY rowid"
        ).fetchall()
