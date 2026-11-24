import faiss
import json
import time
import numpy as np

from pathlib import Path
from datetime import datetime, timezone

from rag.embedding import EMBED_DIM, EMBED_MODEL, embed

LIBRARY = Path("rag/library/library.jsonl")
INDEX = Path("rag/library/index.faiss")

def build_index(library_path: Path = LIBRARY, index_path: Path = INDEX) -> int:

    texts, expected_id = [], 0
    with open(library_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            assert doc["id"] == expected_id
            texts.append(doc["text"])
            expected_id += 1
    if not texts:
        raise ValueError(f"Bibliothèque vide : {library_path}")

    t0 = time.perf_counter()
    vectors = embed(texts)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    faiss.write_index(index, str(index_path))

    metadata_path = index_path.parent / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["index"] = {"model": EMBED_MODEL, "dim": EMBED_DIM, "passages": len(texts), "built_at": datetime.now(timezone.utc).isoformat()}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Index écrit : {len(texts)} passages, {EMBED_DIM} dims | {time.perf_counter() - t0:.0f} s")
    return len(texts)

def search(question: str, k: int = 5, index_path: Path = INDEX) -> list[tuple[int, float]]:

    index = faiss.read_index(str(index_path))
    vector = embed([question])[0]
    k = min(k, index.ntotal)
    scores, ids = index.search(vector.reshape(1,-1).astype(np.float32), k)
    return[(int(i), float(s)) for i, s in zip(ids[0], scores[0])]

if __name__ == "__main__":

    build_index()
    print("\nQuestion témoin :")
    for passage_id, score in search("Comment écrire une fonction en Python ?", k=3):
        print(f" passage {passage_id} - score {score:.3f}")

