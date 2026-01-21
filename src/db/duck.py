import duckdb
import os
from pathlib import Path

DB_PATH = "src/db/results.duckdb"

# Singleton connection for all modules
_conn = None

def get_conn():
    global _conn
    if _conn is None:
        # always read_only in UI
        _conn = duckdb.connect(DB_PATH, read_only=True)
    return _conn
