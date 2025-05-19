import sqlite3
import pandas as pd

# Connect to event.db
conn = sqlite3.connect('event.db')

# Query the drug table
df = pd.read_sql('select name from drug;', conn)

# Print the list of drug names
print("Drug names in event.db:")
print(df['name'].tolist())

# Close the connection
conn.close()