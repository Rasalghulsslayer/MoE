import time
from pathlib import Path

from tokenizer.tokenizer import count_chunks, train_from_counts, build_vocab
from tokenizer.the_regex import REGEX_PATTERN
from utils.runtime import mem_mo

SPLIT_DIR = Path("data/splits")

if __name__ == "__main__":
    print(f"Départ | pic mémoire {mem_mo():.0f} Mo")

    t0 = time.perf_counter()
    counts = count_chunks(SPLIT_DIR / "train_fr.jsonl", REGEX_PATTERN, keep_every=20)
    counts.update(count_chunks(SPLIT_DIR / "train_code.jsonl", REGEX_PATTERN, keep_every=20))
    print(f"Comptage : {time.perf_counter() - t0:.0f} s | {len(counts)} morceaux uniques "
          f"| pic mémoire {mem_mo():.0f} Mo")

    t1 = time.perf_counter()
    merges = train_from_counts(counts, 16_000)
    print(f"Entraînement : {time.perf_counter() - t1:.0f} s | {len(merges)} fusions "
          f"| pic mémoire {mem_mo():.0f} Mo")
    print(f"Total : {time.perf_counter() - t0:.0f} s")

    vocab = build_vocab(merges)
    for idx in list(range(256, 286)) + list(range(len(vocab) - 30, len(vocab))):
        print(f"{idx} : {vocab[idx].decode('utf-8', errors='replace')!r}")

