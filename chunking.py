from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
import os

NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)
nltk.download("punkt", download_dir=NLTK_DIR, quiet=True)
nltk.download("punkt_tab", download_dir=NLTK_DIR, quiet=True)

os.makedirs("chunks", exist_ok=True)
dataset = load_dataset("SamuelYang/bookcorpus", split="train", streaming=True)

chunk_size = 1000
all_sentences = []
chunk_index = 0
total_sentences = 0
max_sentences = 11000

for item in dataset:
    sentences = sent_tokenize(item["text"])
    for s in sentences:
        s = s.strip().replace("\n", " ")
        if not s:
            continue
        all_sentences.append(s)
        total_sentences += 1
        if len(all_sentences) >= chunk_size:
            with open(f"chunks/chunk_{chunk_index}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(all_sentences))
            all_sentences = []
            chunk_index += 1
        if total_sentences >= max_sentences:
            break
    if total_sentences >= max_sentences:
        break

if all_sentences:
    with open(f"chunks/chunk_{chunk_index}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_sentences))

print("OK", total_sentences)
