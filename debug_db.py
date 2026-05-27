import psycopg2
from psycopg2.extras import RealDictCursor

conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/clinical_nlp")
cursor = conn.cursor(cursor_factory=RealDictCursor)

# Test 1: Does cancer exist?
cursor.execute("""
    SELECT concept_id, preferred_term 
    FROM clinical_nlp.concepts 
    WHERE LOWER(preferred_term) LIKE '%cancer%' 
    LIMIT 10
""")
results = cursor.fetchall()
print("=" * 60)
print("TEST 1: Cancer concepts in DB")
print("=" * 60)
if results:
    print(f"✓ Found {len(results)} cancer-related concepts:")
    for row in results:
        print(f"  - {row['concept_id']}: {row['preferred_term']}")
else:
    print("✗ NO CANCER CONCEPTS FOUND")

# Test 2: Total concept count
cursor.execute("SELECT COUNT(*) as cnt FROM clinical_nlp.concepts")
total = cursor.fetchone()['cnt']
print(f"\n✓ Total concepts in DB: {total}")

# Test 3: Embeddings count
cursor.execute("SELECT COUNT(*) as cnt FROM clinical_nlp.embeddings")
emb_total = cursor.fetchone()['cnt']
print(f"✓ Total embeddings: {emb_total}")

conn.close()