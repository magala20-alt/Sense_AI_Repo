import sqlite3

from ..core.config import APP_CONFIG

# This module provides functions to retrieve information about tables in a SQLite database.
def get_table_info(db_path, table_name):
    """Retrieve column names and types for a given table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    conn.close()
    return [{"name": col[1], "type": col[2]} for col in columns]

def get_table_data(db_path, table_name, limit=100):
    """Retrieve data from a given table with an optional limit."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_data(db_path, table_name):
    """Delete all data from a given table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table_name}")
    print(f"Deleted {cursor.rowcount} rows from {table_name}.")
    conn.commit()
    conn.close()

# to run
if __name__ == "__main__":
    info = get_table_info(APP_CONFIG["db_path"], "users")
    data= get_table_data(APP_CONFIG["db_path"], "users")
    print(info)
    print(data)