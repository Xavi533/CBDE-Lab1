import os, glob, time, statistics, chromadb

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="chunks")

files = sorted(glob.glob(os.path.join("chunks", "chunk_*.txt")))
documents, metadatas, ids = [], [], []
i = 0
for fp in files:
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            documents.append(text)
            metadatas.append({"file": os.path.basename(fp), "line": ln})
            ids.append(f"line_{i}")
            i += 1

batch = 1000
times = []
for start in range(0, len(documents), batch):
    end = start + batch
    t0 = time.perf_counter()
    collection.add(documents=documents[start:end], metadatas=metadatas[start:end], ids=ids[start:end])
    t1 = time.perf_counter()
    times.append(t1 - t0)

total = len(documents)
print("Inserted:", total)
print("Batches:", len(times))
print("Min time (s):", round(min(times), 6))
print("Max time (s):", round(max(times), 6))
print("Avg time (s):", round(statistics.mean(times), 6))
print("Std dev (s):", round(statistics.pstdev(times), 6))
