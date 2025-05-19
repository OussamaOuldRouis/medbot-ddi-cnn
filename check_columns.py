import sqlite3

def count_unique_interactions(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(DISTINCT interaction) FROM event;")
    result = cursor.fetchone()
    
    if result:
        print(f"Number of unique interactions: {result[0]}")
    else:
        print("No data found.")

    conn.close()

# Call the function
count_unique_interactions("event.db")
