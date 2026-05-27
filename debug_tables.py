# debug_tables.py
import psycopg2
from psycopg2.extras import RealDictCursor

conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/clinical_nlp")
cursor = conn.cursor(cursor_factory=RealDictCursor)

# List all tables in the database
cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    ORDER BY table_name
""")

tables = cursor.fetchall()
print("=" * 60)
print("ALL TABLES IN DATABASE")
print("=" * 60)

if tables:
    for row in tables:
        print(f"  - {row['table_name']}")
else:
    print("  NO TABLES FOUND")

conn.close()