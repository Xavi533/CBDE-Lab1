import os
import time
import statistics as stats
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("PGHOST")
PORT = int(os.getenv("PGPORT"))
DBNAME = os.getenv("PGDATABASE")
USER = os.getenv("PGUSER")
PASSWORD = os.getenv("PGPASSWORD")
TABLE = "book_sentences"

CHUNKS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "chunks")
BATCH_SIZE = 2000
RUNS = 10

def load_sentences():
    files = [f for f in os.listdir(CHUNKS_DIR) if f.lower().startswith("chunk_") and f.lower().endswith(".txt")]
    files.sort()
    data = []
    for fname in files:
        with open(os.path.join(CHUNKS_DIR, fname), "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                s = line.rstrip("\n")
                if s:
                    data.append((fname, i, s))
    return data

def ms(ns):
    return ns / 1_000_000.0

def main():
    data = load_sentences()
    conn = psycopg2.connect(host=HOST, port=PORT, dbname=DBNAME, user=USER, password=PASSWORD)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
      source text NOT NULL,
      line_index int NOT NULL,
      sentence text NOT NULL,
      emb vector(384),
      PRIMARY KEY (source, line_index)
    )
    """)
    conn.commit()
    times = []
    for _ in range(RUNS):
        cur.execute(f"TRUNCATE TABLE {TABLE}")
        conn.commit()
        t0 = time.perf_counter_ns()
        conn.autocommit = False
        batch = []
        for row in data:
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                execute_values(cur, f"INSERT INTO {TABLE}(source, line_index, sentence) VALUES %s", batch)
                batch = []
        if batch:
            execute_values(cur, f"INSERT INTO {TABLE}(source, line_index, sentence) VALUES %s", batch)
        conn.commit()
        t1 = time.perf_counter_ns()
        times.append(ms(t1 - t0))
    cur.close()
    conn.close()
    print("G0: Storing textual data into postgres")
    print(f"min: {min(times):.3f} ms")
    print(f"max: {max(times):.3f} ms")
    print(f"avg: {stats.mean(times):.3f} ms")
    print(f"std: {stats.pstdev(times) if len(times)>1 else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
