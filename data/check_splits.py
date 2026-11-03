import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from split import text_hash

SPLIT_DIR = Path("data/splits")
SPLITS = ["train", "val", "test"]
DOMAINS = ["fr", "code"]


def check_splits() -> None:
    """Relit les splits produits et vérifie disjonction, couverture et proportions."""
    hashes = defaultdict(set)      # split -> empreintes des textes
    octets = defaultdict(Counter)  # split -> octets par source
    for split in SPLITS:
        for domain in DOMAINS:
            with open(SPLIT_DIR / f"{split}_{domain}.jsonl", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    hashes[split].add(text_hash(doc["text"]))
                    octets[split][doc["source"]] += len(doc["text"].encode("utf-8"))

    for a, b in combinations(SPLITS, 2):
        common = hashes[a] & hashes[b]
        assert not common, f"{len(common)} textes communs entre {a} et {b}"
    print("✓ Splits disjoints")

    sources = sorted(octets["train"])
    for split in ("val", "test"):
        missing = set(sources) - set(octets[split])
        assert not missing, f"Sources absentes de {split} : {missing}"
    print("✓ Toutes les sources sont présentes dans val et test\n")

    print(f"{'source':<25}{'train Mo':>10}{'val %':>8}{'test %':>8}")
    for source in sources:
        total = sum(octets[s][source] for s in SPLITS)
        print(f"{source:<25}{octets['train'][source] / 1e6:>10.1f}"
              f"{100 * octets['val'][source] / total:>8.2f}"
              f"{100 * octets['test'][source] / total:>8.2f}")


if __name__ == "__main__":
    check_splits()
