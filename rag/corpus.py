#Construction de la bibliothèque RAG : les splits de test (jamais vus par les gradients) deviennent des passages.
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from tokenizer.tokenizer import encode_document, load_tokenizer, build_vocab
from tokenizer.special_tokens import BOS_ID, EOS_ID
from rag.chunking import make_chunks

TOKENIZER = Path("artifacts/tokenizer/fr_code_32k.json")
SPLIT_DIR = Path("data/splits")
OUT_DIR = Path("rag/library")

CHUNK_SIZE, OVERLAP, MIN_LEN = 256, 32, 64
SPLITS = ["test_fr", "test_code"]  


def build_split(split: str, merges, pattern, specials, cache: dict, out_f, id_next: int) -> tuple[dict, int]:
    """Transforme un split jsonl en passages, écrits ligne à ligne dans out_f. Les passages abîmés (caractère \ufffd) sont exclus, jamais indexés."""
    n_docs = n_passages = n_tokens = n_fusions = n_exclus = 0
    vocab = build_vocab(merges, specials)
    with open(SPLIT_DIR / f"{split}.jsonl", encoding="utf-8") as src:
        for n_ligne, line in enumerate(src, start=1):
            try:                                        # pipeline à soi : on asserte fort, fichier:ligne dans le message
                doc = json.loads(line)
                text = doc["text"]
                doc_id, source, language = doc["id"], doc["source"], doc.get("language", "?")
                if not text:
                    raise KeyError("text vide")
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"{split}.jsonl:{n_ligne} : document malformé ({e})") from e

            ids = encode_document(text, merges, pattern, cache)
            assert ids[0] == BOS_ID and ids[-1] == EOS_ID, "encode_document doit encadrer le texte de <s> et </s>"
            ids = ids[1:-1]

            passages, stats = make_chunks(ids, vocab, CHUNK_SIZE, OVERLAP, source, MIN_LEN, id_start=id_next)
            for p in passages:
                if "\ufffd" in p["text"]:     # abîmé en amont (docs source) ou é coupé au bord de chunk : on n'indexe pas du demi-bruit
                    n_exclus += 1
                    continue
                p["id"] = id_next                    # renumérotation des survivants : la consécution survit au filtre
                out_f.write(json.dumps({"id": p["id"], "doc_id": doc_id, "source": source,
                                        "language": language, "n_tokens": len(p["token_ids"]),
                                        "text": p["text"]}, ensure_ascii=False) + "\n")
                id_next += 1
                n_passages += 1
                n_tokens += len(p["token_ids"])
            n_docs += 1
            n_fusions += stats["fusions"]
    return {"split": split, "docs": n_docs, "passages": n_passages,
            "tokens": n_tokens, "fusions": n_fusions, "exclus": n_exclus}, id_next


if __name__ == "__main__":
    merges, pattern, specials = load_tokenizer(TOKENIZER)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache, results, id_next = {}, [], 0
    with open(OUT_DIR / "library.jsonl", "w", encoding="utf-8") as out_f:   # une ligne = un passage
        for split in SPLITS:
            t0 = time.perf_counter()
            stats, id_next = build_split(split, merges, pattern, specials, cache, out_f, id_next)
            results.append(stats)
            print(stats, f"| {time.perf_counter() - t0:.0f} s", flush=True)

    metadata = {"tokenizer": str(TOKENIZER), "chunk_size": CHUNK_SIZE, "overlap": OVERLAP,
                "min_len": MIN_LEN, "splits": SPLITS, "regen": "uv run python -m rag.corpus",
                "built_at": datetime.now(timezone.utc).isoformat(), "files": results}
    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        print(f"Bibliothèque écrite : {len(results)} splits, {sum(r['passages'] for r in results)} passages, "
          f"{sum(r['exclus'] for r in results)} exclus (demi-bruit non indexé).", flush=True)