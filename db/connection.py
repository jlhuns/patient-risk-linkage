"""
DATABASE_URL unset -> local SQLite. Set it -> Redshift. Nothing else changes.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent.parent / "warehouse.db"
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")


def get_engine():
    return create_engine(DATABASE_URL)
