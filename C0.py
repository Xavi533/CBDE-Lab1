import os
import glob
import chromadb
from chromadb.config import Settings

def read_chunks(folder):
    files = sorted(glob.glob(os.path.join(folder, "chunk_*.txt")))
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
            yield os.path.basename(path), lines

def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    data_dir = "chunks"
    persist_dir = "./chroma_db"
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
    collection = client.get_or_create_collection(name="bookcorpus_chunks", metadata={"source": "chunks_txt"})
    for fname, sentences in read_chunks(data_dir):
        ids = [f"{fname}:{i}" for i in range(len(sentences))]
        metadatas = [{"file": fname, "line": i} for i in range(len(sentences))]
        for b_ids, b_docs, b_meta in zip(batched(ids, 1000), batched(sentences, 1000), batched(metadatas, 1000)):
            collection.add(ids=b_ids, documents=b_docs, metadatas=b_meta)
    count = collection.count()
    print(count)

if __name__ == "__main__":
    main()
