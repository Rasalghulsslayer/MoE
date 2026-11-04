#Script d'entraînement du tokenizeur.
import time
from datetime import datetime, timezone
from pathlib import Path

from tokenizer.special_tokens import SPECIALS, SPECIALS_START, VOCAB_SIZE
from tokenizer.the_regex import REGEX_PATTERN
from tokenizer.tokenizer import count_chunks, save_tokenizer, train_from_counts
from utils.runtime import mem_mo

SPLIT_DIR = Path("data/splits")
OUT = Path("artifacts/tokenizer/fr_code_32k.json")
KEEP_EVERY = 5

if __name__ == "__main__":
    t0 = time.perf_counter()
    counts = count_chunks(SPLIT_DIR / "train_fr.jsonl", REGEX_PATTERN, keep_every=KEEP_EVERY)
    counts.update(count_chunks(SPLIT_DIR / "train_code.jsonl", REGEX_PATTERN, keep_every=KEEP_EVERY))
    print(f"Comptage : {time.perf_counter() - t0:.0f} s | {len(counts)} morceaux uniques "
          f"| pic mémoire {mem_mo():.0f} Mo", flush=True)

    merges = train_from_counts(counts, SPECIALS_START)
    elapsed = time.perf_counter() - t0
    print(f"Entraînement : {elapsed:.0f} s | {len(merges)} fusions | pic mémoire {mem_mo():.0f} Mo")

    save_tokenizer(OUT, merges, REGEX_PATTERN, SPECIALS, {
        "train_files": ["train_fr.jsonl", "train_code.jsonl"],
        "keep_every": KEEP_EVERY,
        "vocab_size": VOCAB_SIZE,
        "unique_chunks": len(counts),
        "train_seconds": round(elapsed),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Tokenizer enregistré dans {OUT}")
