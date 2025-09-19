import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def main():
    persist_dir = "./chroma_db"
    collection_name = "bookcorpus_chunks"
    batch_size = 1000
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
    collection = client.get_or_create_collection(name=collection_name, metadata={"source": "chunks_txt"})
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    offset = 0
    while True:
        res = collection.get(include=["documents"], limit=batch_size, offset=offset)
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        if not ids:
            break
        emb = model.encode(docs)
        emb = [e.tolist() for e in emb]
        collection.update(ids=ids, embeddings=emb)
        offset += len(ids)
    print(collection.count())

if __name__ == "__main__":
    main()
