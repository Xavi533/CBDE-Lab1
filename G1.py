import os, time, statistics as stats
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


HOST = os.getenv("PGHOST")
PORT = int(os.getenv("PGPORT"))
DBNAME = os.getenv("PGDATABASE")
USER = os.getenv("PGUSER")
PASSWORD = os.getenv("PGPASSWORD")
TABLE = "book_sentences"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384
BATCH_ENCODE = 1024
BATCH_DB = 2000
RUNS = 10

def ms(ns): return ns/1_000_000.0
def to_vec_literal(a): return "[" + ",".join(f"{x:.6f}" for x in a) + "]"

def main():
    conn = psycopg2.connect(host=HOST, port=PORT, dbname=DBNAME, user=USER, password=PASSWORD)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE}(
        source text NOT NULL,
        line_index int NOT NULL,
        sentence text NOT NULL,
        emb vector({EMB_DIM}),
        PRIMARY KEY(source,line_index)
    )""")
    conn.commit()
    cur.execute(f"SELECT source,line_index,sentence FROM {TABLE} ORDER BY source,line_index")
    rows = cur.fetchall()
    sentences = [r[2] for r in rows]
    model = SentenceTransformer(MODEL_NAME)
    embs = []
    for i in range(0, len(sentences), BATCH_ENCODE):
        embs.append(model.encode(sentences[i:i+BATCH_ENCODE], convert_to_numpy=True, normalize_embeddings=False))
    embs = np.vstack(embs) if embs else np.zeros((0, EMB_DIM), dtype=np.float32)
    triples = [(rows[i][0], rows[i][1], to_vec_literal(embs[i])) for i in range(len(rows))]
    times = []
    for _ in range(RUNS):
        cur.execute(f"UPDATE {TABLE} SET emb=NULL")
        conn.commit()
        t0 = time.perf_counter_ns()
        batch = []
        for rec in triples:
            batch.append(rec)
            if len(batch) >= BATCH_DB:
                execute_values(cur, f"UPDATE {TABLE} AS t SET emb=v.emb::vector FROM (VALUES %s) AS v(source,line_index,emb) WHERE t.source=v.source AND t.line_index=v.line_index", batch)
                batch = []
        if batch:
            execute_values(cur, f"UPDATE {TABLE} AS t SET emb=v.emb::vector FROM (VALUES %s) AS v(source,line_index,emb) WHERE t.source=v.source AND t.line_index=v.line_index", batch)
        conn.commit()
        t1 = time.perf_counter_ns()
        times.append(ms(t1 - t0))
    cur.close(); conn.close()
    print("G1: storing the embeddings")
    print(f"min: {min(times):.3f} ms")
    print(f"max: {max(times):.3f} ms")
    print(f"avg: {stats.mean(times):.3f} ms")
    print(f"std: {stats.pstdev(times) if len(times)>1 else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
