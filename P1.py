import os
import psycopg2
from sentence_transformers import SentenceTransformer

HOST = os.getenv("PGHOST")
PORT = int(os.getenv("PGPORT"))
DBNAME = os.getenv("PGDATABASE")
USER = os.getenv("PGUSER")
PASSWORD = os.getenv("PGPASSWORD")
TABLE = "book_sentences"

conn = psycopg2.connect(host=HOST, port=PORT, dbname=DBNAME, user=USER, password=PASSWORD)
conn.autocommit = True
cur = conn.cursor()
cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS embedding DOUBLE PRECISION[]")
cur.execute(f"SELECT id, sentence FROM {TABLE} ORDER BY id")
rows = cur.fetchall()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
n = 0
for _id, text in rows:
    emb = model.encode([text], normalize_embeddings=False, convert_to_numpy=True)[0].tolist()
    cur.execute(f"UPDATE {TABLE} SET embedding=%s WHERE id=%s", (emb, _id))
    n += 1
cur.close()
conn.close()
print(n)
