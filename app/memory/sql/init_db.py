"""Initialize the SQLite trading database from schema.sql.

Run:  ./venv/bin/python -m app.memory.sql.init_db
Idempotent — every table uses CREATE TABLE IF NOT EXISTS, so re-running is safe.
"""

import sqlite3
import sys
from pathlib import Path

# Make `app` importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.config import settings


def initialize_database() -> None:
    db_file = settings.sql_db_path
    schema_file = settings.sql_schema_path

    if not schema_file.exists():
        print(f"❌ Error: could not find schema at {schema_file}")
        return

    db_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"1. Connecting to SQLite at {db_file} ...")
    conn = sqlite3.connect(str(db_file))
    print("2. Executing schema ...")
    try:
        conn.executescript(schema_file.read_text(encoding="utf-8"))
        print(f"✅ Success! Database initialized at {db_file}")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Error executing schema: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    initialize_database()
