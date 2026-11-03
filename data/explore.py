#Exploration du dataset
from datasets import load_dataset
from datasets import get_dataset_config_names
from collections import Counter
import json
from pathlib import Path
from utils.runtime import mem_mo

DATASET = "OpenLLM-France/Lucie-Training-Dataset"

def peek(config: str, n: int = 3) -> None:
    """Affiche les n premiers documents d'un sous-ensemble."""
    ds = load_dataset(DATASET, config, split="train", streaming=True)
    print(f"\n===== {config} =====")
    for i, doc in enumerate(ds.take(n)):
        print(f"\n--- document {i} ---")
        for key, value in doc.items():
            if key == "text":
                print(f"text ({len(value)} caractères) : {value[:300]!r}")
            else :
                print(f"{key} : {value!r}")

def source_distribution(config: str, n: int=1000, shuffle: bool = False) -> None:
    """Compte documents et octets par source sur les n premiers documents."""
    ds = load_dataset(DATASET, config, split="train", streaming=True, columns=["id", "source", "text"])
    if shuffle:
        ds = ds.shuffle(seed=42, buffer_size=10000)
    docs, octets = Counter(), Counter()
    for i, doc in enumerate(ds.take(n)):
        docs[doc["source"]] += 1
        octets[doc["source"]] += len(doc["text"].encode("utf-8"))
        if i % 100 == 0:
            print(f"doc {i} | {doc['source']} | {len(doc['text'])} car. | pic mémoire {mem_mo():.0f} Mo", flush=True)
    print(f"\n===== {config} | shuffle={shuffle} =====")
    for source, count in docs.most_common():
        print(f"{source:<25} {count:>5} docs {octets[source] / 1e6:>8.2f} Mo")

def list_configs() -> None:
    """Affiche toutes les configs disponibles dans la version v1.2."""
    configs = get_dataset_config_names(DATASET, revision="v1.2")
    print(f"{len(configs)} configs")
    for name in configs:
        print(name)

def HAL_check() -> None:
    ds = load_dataset(DATASET, "HAL", split ="train", streaming=True)
    for i, doc in enumerate(ds.take(5)):
        print(f" Document n°{i} \n")
        for key, value in doc.items():
            print(f"{key} : {value!r}")

def top_duplicates(path: Path, k: int = 15) -> None:
    """Affiche les textes les plus répétés d'un fichier JSONL."""
    counts = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            counts[json.loads(line)["text"]] += 1
    for text, n in counts.most_common(k):
        print(f"{n:>5} × {text[:80]!r}")


if __name__ == "__main__":
    #peek("fr")
    #source_distribution("fr", shuffle=True) impossible due to document size
    #list_configs()
    #HAL_check()
    top_duplicates(Path("data/raw/fr/InterventionsParlement-000.jsonl"))
