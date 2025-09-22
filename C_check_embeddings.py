import chromadb
from chromadb.config import Settings

persist_dir = "./chroma_db"
collection_name = "chunks"
batch_size = 1000

client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
col = client.get_or_create_collection(collection_name)

total = col.count()
offset = 0
with_emb = 0
emb_dim = None
missing = []

while True:
    res = col.get(include=["embeddings"], limit=batch_size, offset=offset)
    ids = res.get("ids", [])
    embs = res.get("embeddings", [])
    if not ids:
        break
    for i, e in enumerate(embs):
        if e is not None:
            with_emb += 1
            if emb_dim is None:
                emb_dim = len(e)
        else:
            if len(missing) < 20:
                missing.append(ids[i])
    offset += len(ids)

print("total_items:", total)
print("items_with_embeddings:", with_emb)
print("all_embedded:", with_emb == total)
print("embedding_dim:", emb_dim)
print("sample_missing_ids:", missing)
