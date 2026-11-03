import hashlib
import json
from collections import defaultdict
from pathlib import Path
from collections import Counter

RAW_DIR = Path("data/raw")
SPLIT_DIR = Path("data/splits")
DOMAINS = ["fr", "code"]
EVAL_FRAC = 0.01

def text_hash(text: str) -> str:
    """Empreinte déterministe du contenu d'un document."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def assign_splits() -> dict[tuple[str, str], str]:
    """Passe 1 : on choisit le split de chaque document, stratifié par source."""
    by_source = defaultdict(list)
    seen = set()
    duplicates = Counter()
    for domain in DOMAINS:
        for path in sorted((RAW_DIR / domain).glob("*.jsonl")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    h = text_hash(doc["text"])
                    if h in seen:
                        duplicates[doc["source"]] += 1
                        continue
                    seen.add(h)
                    by_source[doc["source"]].append(h)
    print("Doublons exact par source :", duplicates)
    assignment = {}
    for source, hashes in by_source.items():
        hashes.sort()
        n_eval = max(1, round(len(hashes) * EVAL_FRAC))
        for i, h in enumerate(hashes):
            if i < n_eval:
                assignment[h] = "val"
            elif i < 2 * n_eval:
                assignment[h] = "test"
            else:
                assignment[h] = "train"
    return assignment

def write_split(assignment: dict[tuple[str, str], str]) -> None:
    """Passe 2: Recopie chaque document dans le dossier de son split."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    written = set()
    for domain in DOMAINS:
        outs = {split:open(SPLIT_DIR / f"{split}_{domain}.jsonl", "w", encoding="utf-8")
                for split in ("train", "val", "test")}
        try:
            for path in sorted((RAW_DIR / domain).glob("*.jsonl")):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        doc = json.loads(line)
                        h = text_hash(doc["text"])
                        if h in written:
                            continue
                        written.add(h)
                        outs[assignment[h]].write(line)
        finally:
            for out in outs.values():
                out.close()

if __name__ == "__main__":
    assignment = assign_splits()
    counts = Counter(assignment.values())
    print(f"Répartition des splits : {counts}")
    write_split(assignment)

