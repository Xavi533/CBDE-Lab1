import os, json, time
import numpy as np
import chromadb
from chromadb.config import Settings

persist_dir = "./chroma_db"
collection_name = "chunks"
batch_size = 1000

client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
col = client.get_or_create_collection(collection_name)

all_ids = []
all_docs = []
all_embs = []

offset = 0
while True:
    res = col.get(include=["embeddings", "documents"], limit=batch_size, offset=offset)
    ids = res.get("ids", [])
    docs = res.get("documents", [])
    embs = res.get("embeddings", [])
    if not ids:
        break
    for i in range(len(ids)):
        if embs[i] is not None:
            all_ids.append(ids[i])
            all_docs.append(docs[i])
            all_embs.append(embs[i])
    offset += len(ids)

order = np.argsort(np.array(all_ids, dtype=object))
ids = np.array([all_ids[i] for i in order], dtype=object)
texts = [all_docs[i] for i in order]
X = np.array([all_embs[i] for i in order], dtype=float)
norms = np.linalg.norm(X, axis=1)
chosen = list(zip(ids[:10], texts[:10]))

def top2_cosine(qv):
    qn = np.linalg.norm(qv)
    sims = (X @ qv) / (norms * qn + 1e-12)
    return sims

def top2_euclidean(qv):
    dif = X - qv
    d = np.sqrt(np.sum(dif * dif, axis=1))
    return d

results = []
times_cos = []
times_euc = []

for qid, qtext in chosen:
    idx = int(np.where(ids == qid)[0][0])
    qv = X[idx]
    t0 = time.perf_counter()
    sims = top2_cosine(qv)
    sims[idx] = -np.inf
    top_idx = np.argpartition(-sims, 2)[:2]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    t1 = time.perf_counter()
    times_cos.append((t1 - t0) * 1000.0)
    t0 = time.perf_counter()
    dists = top2_euclidean(qv)
    dists[idx] = np.inf
    top_idx_e = np.argpartition(dists, 2)[:2]
    top_idx_e = top_idx_e[np.argsort(dists[top_idx_e])]
    t1 = time.perf_counter()
    times_euc.append((t1 - t0) * 1000.0)
    res = {
        "query_id": str(qid),
        "query_sentence": qtext,
        "top2": {
            "cosine": [
                {"id": str(ids[top_idx[0]]), "similarity": float(sims[top_idx[0]]), "sentence": texts[top_idx[0]]},
                {"id": str(ids[top_idx[1]]), "similarity": float(sims[top_idx[1]]), "sentence": texts[top_idx[1]]}
            ],
            "euclidean": [
                {"id": str(ids[top_idx_e[0]]), "distance": float(dists[top_idx_e[0]]), "sentence": texts[top_idx_e[0]]},
                {"id": str(ids[top_idx_e[1]]), "distance": float(dists[top_idx_e[1]]), "sentence": texts[top_idx_e[1]]}
            ]
        }
    }
    results.append(res)

def stats(a):
    a = np.array(a, dtype=float)
    return {"min": float(a.min()), "max": float(a.max()), "avg": float(a.mean()), "std": float(a.std(ddof=0))}

out = {
    "selected": [{"id": str(qid), "sentence": qtext} for qid, qtext in chosen],
    "results": results,
    "timing_summary_ms": {"cosine": stats(times_cos), "euclidean": stats(times_euc)}
}

print(json.dumps(out, ensure_ascii=False, indent=2))
with open("c2_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)