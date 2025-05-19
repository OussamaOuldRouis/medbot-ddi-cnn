import sqlite3

def discover_schema(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    if not tables:
        print("No tables found in the database.")
        return

    print(f"Schema for database: {db_path}")
    for table_name in tables:
        table = table_name[0]
        print(f"\nTable: {table}")
        # Get column info for each table
        cursor.execute(f"PRAGMA table_info('{table}')")
        columns = cursor.fetchall()
        for col in columns:
            cid, name, col_type, notnull, dflt_value, pk = col
            print(f"  Column: {name} | Type: {col_type} | Not Null: {bool(notnull)} | Default: {dflt_value} | Primary Key: {bool(pk)}")

    # Close the connection
    conn.close()

# Call the function with the path to your database
discover_schema("event.db")
