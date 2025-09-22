import os, time, json, statistics as stats
import psycopg2
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("PGHOST")
PORT = int(os.getenv("PGPORT"))
DBNAME = os.getenv("PGDATABASE")
USER = os.getenv("PGUSER")
PASSWORD = os.getenv("PGPASSWORD")
TABLE = "book_sentences"
K = 2
N_QUERIES = 10
OUTFILE = "p2_top2.json"

def ms(ns): return ns/1_000_000.0

def main():
    conn = psycopg2.connect(host=HOST, port=PORT, dbname=DBNAME, user=USER, password=PASSWORD)
    cur = conn.cursor()
    cur.execute(f"SELECT source,line_index FROM {TABLE} WHERE emb IS NOT NULL ORDER BY source,line_index LIMIT %s", (N_QUERIES,))
    seeds = cur.fetchall()
    if len(seeds) < N_QUERIES:
        raise SystemExit("Not enough embedded rows.")
    results = []
    t_cos, t_l2 = [], []
    for s, i in seeds:
        cur.execute(f"SELECT sentence FROM {TABLE} WHERE source=%s AND line_index=%s", (s, i))
        q_sent = cur.fetchone()[0]
        t0 = time.perf_counter_ns()
        cur.execute(
            f"""
            SELECT t.source,t.line_index,t.sentence,cosine_distance(t.emb,q.emb) AS d
            FROM {TABLE} t,(SELECT emb FROM {TABLE} WHERE source=%s AND line_index=%s) q
            WHERE t.emb IS NOT NULL AND NOT (t.source=%s AND t.line_index=%s)
            ORDER BY d
            LIMIT %s
            """,
            (s, i, s, i, K),
        )
        cos_rows = cur.fetchall()
        t1 = time.perf_counter_ns()
        t_cos.append(ms(t1 - t0))
        t0 = time.perf_counter_ns()
        cur.execute(
            f"""
            SELECT t.source,t.line_index,t.sentence,l2_distance(t.emb,q.emb) AS d
            FROM {TABLE} t,(SELECT emb FROM {TABLE} WHERE source=%s AND line_index=%s) q
            WHERE t.emb IS NOT NULL AND NOT (t.source=%s AND t.line_index=%s)
            ORDER BY d
            LIMIT %s
            """,
            (s, i, s, i, K),
        )
        l2_rows = cur.fetchall()
        t1 = time.perf_counter_ns()
        t_l2.append(ms(t1 - t0))
        results.append({
            "query": {"source": s, "line_index": i, "sentence": q_sent},
            "cosine_top2": [{"source": r[0], "line_index": r[1], "sentence": r[2], "distance": float(r[3])} for r in cos_rows],
            "euclidean_top2": [{"source": r[0], "line_index": r[1], "sentence": r[2], "distance": float(r[3])} for r in l2_rows],
        })
    cur.close(); conn.close()
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    print("G2: top-2 similar")
    print("Cosine:")
    print(f"min: {min(t_cos):.3f} ms")
    print(f"max: {max(t_cos):.3f} ms")
    print(f"avg: {stats.mean(t_cos):.3f} ms")
    print(f"std: {stats.pstdev(t_cos) if len(t_cos)>1 else 0.0:.3f} ms")
    print()
    print("Euclidean:")
    print(f"min: {min(t_l2):.3f} ms")
    print(f"max: {max(t_l2):.3f} ms")
    print(f"avg: {stats.mean(t_l2):.3f} ms")
    print(f"std: {stats.pstdev(t_l2) if len(t_l2)>1 else 0.0:.3f} ms")
    print(f"\nSaved: {OUTFILE}")

if __name__ == "__main__":
    main()
