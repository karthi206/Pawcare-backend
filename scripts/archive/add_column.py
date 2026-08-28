import sqlite3

conn = sqlite3.connect('instance/cases.db')
cursor = conn.cursor()
try:
    cursor.execute('ALTER TABLE "case" ADD COLUMN vet_confirmed_label VARCHAR(100)')
    conn.commit()
    print("Column added successfully.")
except sqlite3.OperationalError as e:
    print(f"Skipped (likely already exists): {e}")
conn.close()