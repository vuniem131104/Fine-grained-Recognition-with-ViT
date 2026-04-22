"""Run Alembic migrations to create/upgrade all tables."""
from __future__ import annotations

import sys

from alembic import command
from alembic.config import Config


def main() -> None:
    cfg = Config("common/database/alembic.ini")
    command.upgrade(cfg, "head")
    print("All tables are ready.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
