#Encodage du corpus pour ingestion du modèle.
import json
import time
from pathlib import Path

import numpy as np

from tokenizer.tokenizer import encode_document, load_tokenizer

TOKENIZER = Path("artifacts/tokenizer/fr_code_32k.json")
SPLIT_DIR = Path("data/splits")
OUT_DIR = Path("data/tokens")
FLUSH_EVERY = 10_000_000  # nombre de tokens gardés en mémoire avant écriture


def encode_split(name: str, merges, pattern, cache: dict) -> dict:
    """Encode un split JSONL en fichier binaire uint16, un document après l'autre."""
    n_docs = n_tokens = n_bytes = 0
    buffer = []
    with open(SPLIT_DIR / f"{name}.jsonl", encoding="utf-8") as src, \
         open(OUT_DIR / f"{name}.bin", "wb") as dst:
        for line in src:
            text = json.loads(line)["text"]
            buffer.extend(encode_document(text, merges, pattern, cache))
            n_docs += 1
            n_bytes += len(text.encode("utf-8"))
            if len(buffer) >= FLUSH_EVERY:
                np.array(buffer, dtype=np.uint16).tofile(dst)
                n_tokens += len(buffer)
                buffer = []
        np.array(buffer, dtype=np.uint16).tofile(dst)
        n_tokens += len(buffer)
    text_tokens = n_tokens - 2 * n_docs  # sans les <s> et </s>
    return {"split": name, "docs": n_docs, "tokens": n_tokens,
            "bytes_per_token": round(n_bytes / text_tokens, 3)}


if __name__ == "__main__":
    merges, pattern, specials = load_tokenizer(TOKENIZER)
    assert max(specials.values()) < 2**16, "Les ids ne tiennent pas en uint16"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    results = []
    for name in ["val_fr", "val_code", "test_fr", "test_code", "train_fr", "train_code"]:
        t0 = time.perf_counter()
        results.append(encode_split(name, merges, pattern, cache))
        print(results[-1], f"| {time.perf_counter() - t0:.0f} s", flush=True)
    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"tokenizer": str(TOKENIZER), "splits": results}, f, indent=2)
