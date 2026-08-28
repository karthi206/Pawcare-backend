import sqlite3

conn = sqlite3.connect('instance/cases.db')
cursor = conn.cursor()
try:
    cursor.execute('ALTER TABLE "case" ADD COLUMN reviewed_by_id INTEGER')
    conn.commit()
    print("Column added successfully.")
except sqlite3.OperationalError as e:
    print(f"Skipped (likely already exists): {e}")
conn.close()