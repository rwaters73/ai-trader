# tools/migrate_db_add_columns.py

# tools/migrate_db_add_columns.py

import os
import sys
import sqlite3

# -----------------------------------------------------
# Ensure project root is on the import path
# -----------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from db import DB_PATH


def column_exists(connection: sqlite3.Connection, table: str, column: str) -> bool:
    cursor = connection.execute(f"PRAGMA table_info({table})")
    rows = cursor.fetchall()
    return any(row[1] == column for row in rows)


def main() -> None:
    print(f"Opening database at {DB_PATH}...")
    connection = sqlite3.connect(DB_PATH)

    try:
        # Columns we want to be sure exist on the orders table
        columns_to_add = [
            ("orders", "limit_price", "REAL"),
            ("orders", "stop_price", "REAL"),
        ]

        for table, column, column_type in columns_to_add:
            if column_exists(connection, table, column):
                print(f"[OK] {table}.{column} already exists")
            else:
                print(f"[MIGRATE] Adding column {table}.{column} {column_type} ...")
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
                )
                print(f"[DONE] Added {table}.{column}")

        connection.commit()
        print("Migration complete.")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
