import os, glob
import psycopg2
from psycopg2.extras import execute_values

CHUNKS_DIR = "chunks"
MAX_ROWS = 11000
HOST = os.getenv("PGHOST")
PORT = int(os.getenv("PGPORT"))
DBNAME = os.getenv("PGDATABASE")
USER = os.getenv("PGUSER")
PASSWORD = os.getenv("PGPASSWORD")
TABLE = "book_sentences"

files = sorted(glob.glob(os.path.join(CHUNKS_DIR, "chunk_*.txt")))
rows = []
for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if s:
                rows.append((os.path.basename(fp), i, s))
                if len(rows) >= MAX_ROWS:
                    break
    if len(rows) >= MAX_ROWS:
        break

conn = psycopg2.connect(host=HOST, port=PORT, dbname=DBNAME, user=USER, password=PASSWORD)
conn.autocommit = True
cur = conn.cursor()
cur.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} (id BIGSERIAL PRIMARY KEY, source_file TEXT, line_no INT, sentence TEXT NOT NULL)")
cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS source_file TEXT")
cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS line_no INT")
cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS sentence TEXT")
if rows:
    execute_values(cur, f"INSERT INTO {TABLE} (source_file, line_no, sentence) VALUES %s", rows, page_size=1000)
cur.close()
conn.close()
print(len(rows))
