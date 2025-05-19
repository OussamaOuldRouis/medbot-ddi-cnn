import sqlite3
import pandas as pd
from supabase import create_client

# Connect to event.db
conn = sqlite3.connect('event.db')

# Query the event table to get drug interactions
df_interactions = pd.read_sql('''
    SELECT name1, name2, interaction
    FROM event;
''', conn)

# Close the connection
conn.close()

# Connect to Supabase
supabase_url = "https://nhacxhyjnrkrhbhrjcmz.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5oYWN4aHlqbnJrcmhiaHJqY216Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU5OTM4NzIsImV4cCI6MjA2MTU2OTg3Mn0.ZWb1eTozqxCTDh8il0Dzwb4TCFY6xxFAKQeR1qoqeFM"
supabase = create_client(supabase_url, supabase_key)

# Insert drug interactions into Supabase
for _, row in df_interactions.iterrows():
    supabase.table('drug_interactions').insert({
        'drug1': row['name1'],
        'drug2': row['name2'],
        'interaction_description': row['interaction'],
        'severity': 'moderate',  # Default severity, adjust as needed
        'is_approved': True
    }).execute()

print("Drug interactions inserted into Supabase.")