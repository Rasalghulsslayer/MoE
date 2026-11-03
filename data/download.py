#Téléchargement du dataset.
import json
from pathlib import Path
from datasets import load_dataset
from datetime import datetime, timezone 

DATASET = "OpenLLM-France/Lucie-Training-Dataset"
REVISION = "v1.2"
COLUMNS = ["id", "source", "language", "text"]
SHARD_BYTES = 100000000 # taille cible d'un fichier (100Mo)
FR_MIX = {
    "RedPajama-fr": 0.70,
    "Wikipedia-fr": 0.10,
    "GallicaPress": 0.05,
    "GallicaMonographies": 0.025,
    "Gutenberg-fr": 0.025,
    "HAL": 0.025,
    "Theses": 0.025,
    "Claire-fr": 0.025,
    "InterventionsParlement": 0.025,
}
FR_BUDGET = 800000000
CODE_BUDGET = 200000000
RAW_DIR = Path("data/raw")

def download_source(config: str, budget: int, out_dir: Path, lang: str | None = None) -> dict:
    """Écrit les documents de 'config' en JSON jusqu'à 'budget' octets de texte."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET, config, split="train", streaming=True, revision=REVISION, columns=COLUMNS)
    total = n_docs = skipped = shard = shard_bytes = 0
    f = open(out_dir / f"{config}-{shard:03d}.jsonl", "w", encoding="utf-8")
    for doc in ds:
        if lang is not None and doc["language"] != lang:
            skipped +=1
            continue
        size = len(doc["text"].encode("utf-8"))
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        total += size
        n_docs += 1
        shard_bytes += size
        if total >= budget:
            break
        if shard_bytes >= SHARD_BYTES:
            f.close()
            shard += 1 
            shard_bytes = 0
            f = open(out_dir / f"{config}-{shard:03d}.jsonl", "w", encoding="utf-8")
    f.close()
    return {"config": config, "budget": budget, "docs": n_docs, "bytes": total, "skipped": skipped, "shards" : shard + 1}

def download_all() -> None:
    """Télécharge le mélange français et le code, puis écrit les métadonnées"""
    assert abs(sum(FR_MIX.values()) - 1.0) < 1e-9
    results = []
    for config, share in FR_MIX.items():
        print(f" {config}-{share: .1%}", flush=True) 
        results.append(download_source(config, int(share * FR_BUDGET), RAW_DIR / "fr", lang="fr"))
    print("-> code-python", flush=True)
    results.append(download_source("code-python", CODE_BUDGET, RAW_DIR / "CODE"))
    metadata = {
        "dataset": DATASET,
        "revision": REVISION,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "sources": results,
    }
    with open(RAW_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    download_all()