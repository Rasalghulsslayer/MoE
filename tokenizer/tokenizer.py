#Tokenizer basics
from collections import Counter, defaultdict
import unicodedata
import regex
import json
from pathlib import Path
from tokenizer.the_regex import REGEX_PATTERN
from tokenizer.tries.try_regex import GPT4_PATTERN
from tokenizer.special_tokens import BOS_ID, EOS_ID 



def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Compte le nombre d'occurence de chaque paire de tokens successifs."""
    stats = Counter(zip(ids, ids[1:]))
    #for (id_1, id_2) in zip(ids, ids[1:]):
        #stats[(id_1, id_2)] += 1
    return stats


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Transforme la liste d'identifiant initiale en fusionnant la paire indiquée."""
    merged = []
    i = 0
    while i < len(ids) - 1:
        if (ids[i], ids[i+1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    if i == len(ids) - 1:
        merged.append(ids[-1])
    return merged


def train(text: str, vocab_size: int) -> dict[tuple[int, int], int]:
    """Entrainement du tokenizer sur le texte, alliant les paires les plus redondantes."""
    assert vocab_size >= 256, "Variable vocab_size non valide, proposez plus grand (>= 256)"
    ids = list(text.encode("utf-8"))
    merges = {}
    num_merges = vocab_size - 256
    for i in range(num_merges):
        stats = get_stats(ids)
        new_id = 256 + i
        if not stats:
            break
        pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1])) #pair = max(stats, key=stats.get) version non optimisée
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
    return merges


def train_reference(text: str, vocab_size: int, pattern=REGEX_PATTERN) -> dict[tuple[int, int], int]:
    """Entrainement du tokenizer sur le texte, alliant les paires les plus redondantes, version impliquant regex."""
    assert vocab_size >= 256, "Variable vocab_size non valide, proposez plus grand (>= 256)"

    text_chunks = pretokenize(text, pattern)

    counts = Counter(text_chunks)
    words = [list(chunk.encode('utf-8')) for chunk in counts]
    freqs = list(counts.values())
 
    merges = {}
    num_merges = vocab_size - 256

    for i in range(num_merges):
        stats = Counter()
        for (word, freq) in zip(words, freqs):
            for pair in zip(word, word[1:]):
                stats[pair] += freq
        new_id = 256 + i
        if not stats:
            break
        pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
        words = [merge(word, pair, new_id) for word in words]
        merges[pair] = new_id
    return merges


def train_from_counts(counts, vocab_size):
    """Pour l'instant je ne sais pas."""
    words = [list(chunk.encode('utf-8')) for chunk in counts]
    freqs = list(counts.values())
    stats = Counter()
    where = defaultdict(set)
    for j, (word, freq) in enumerate(zip(words, freqs)):
        for pair in zip(word, word[1:]):
            stats[pair] += freq
            where[pair].add(j) 
    
    merges = {}
    num_merges = vocab_size - 256

    for i in range(num_merges):
        new_id = 256 + i
        if not stats:
            break
        pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1])) #pair = max(stats, key=stats.get) version non optimisée
        for j in where.pop(pair):
            for remove_pair in zip(words[j], words[j][1:]):
                stats[remove_pair] -= freqs[j]
                if stats[remove_pair] == 0: 
                    stats.pop(remove_pair, None)
            words[j] = merge(words[j], pair, new_id)
            for new_pair in zip(words[j], words[j][1:]):
                stats[new_pair] += freqs[j]
                where[new_pair].add(j)
        assert pair not in stats, f"La paire {pair} n'a pas été entièrement retirée, il reste {stats[pair]}"
        merges[pair] = new_id
    return merges


def train_reg(text: str, vocab_size: int, pattern=REGEX_PATTERN) -> dict[tuple[int, int], int]:
    """Entrainement du tokenizer sur le texte, alliant les paires les plus redondantes, version impliquant regex, optimisée"""
    assert vocab_size >= 256, "Variable vocab_size non valide, proposez plus grand (>= 256)"

    text_chunks = pretokenize(text, pattern)

    counts = Counter(text_chunks)

    return train_from_counts(counts, vocab_size)



def count_chunks(path, pattern=REGEX_PATTERN, keep_every = 20) -> Counter:
    """Pareil je ne saurais pas trop dire."""
    counts = Counter()
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % keep_every == 0:
                doc = json.loads(line)["text"]
                counts.update(pretokenize(doc, pattern))
            else:
                continue
    return counts
    

def build_vocab(merges: dict[tuple[int, int], int], specials: dict[str, int] | None = None) -> dict[int, bytes]:
    """Construit le vocabulaire à partir de la liste de fusions et des caractères spéciaux."""
    vocab = {}
    for i in range (256):
        vocab[i] = bytes([i])
    for (a, b), i in (merges.items()):
        vocab[i] = (vocab[a] + vocab[b])
    for name, idx in (specials or {}).items():
        vocab[idx] = name.encode('utf-8')
    return vocab


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """Restitue la liste d'indices en phrase."""
    output_l = []
    for token in ids:
        output_l.append(vocab[token])
    output_b = b''.join(output_l)
    output_s = output_b.decode('utf-8', errors="replace")
    return output_s


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """Transforme le texte en liste d'indices."""
    ids = list(text.encode('utf-8'))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def encode_chunks(text: str, merges: dict[tuple[int, int], int], pattern=REGEX_PATTERN) -> list[int]:
    """Encode pour les chunks."""
    ids = []
    chunks = pretokenize(text, pattern)
    for chunk in chunks:
        id = encode(chunk, merges)
        ids += id
    return ids


def encode_document(text: str, merges: dict[tuple[int, int], int], pattern: str | None = REGEX_PATTERN) -> list[int]:
    """Encode un document complet, encadré par les tokens de début et de fin."""
    return [BOS_ID] + encode_chunks(text, merges, pattern) + [EOS_ID]



def pretokenize(text, pattern):
    if pattern == None:
        return [text]
    else:
        return regex.findall(pattern, unicodedata.normalize("NFC", text))


FORMAT_VERSION = 1


def save_tokenizer(path: Path, merges: dict[tuple[int, int], int], pattern: str | None, specials: dict[str, int], metadata: dict) -> None:
    """Enregistre le tokenizer en JSON : fusions ordonnées, pré-tokenisation, tokens spéciaux, métadonnées."""
    ordered = sorted(merges.items(), key=lambda item: item[1])
    assert [idx for _, idx in ordered] == list(range(256, 256 + len(ordered))), "Ids de fusions non contigus"
    data = {
        "format_version": FORMAT_VERSION,
        "normalization": "NFC",
        "pattern": pattern,
        "merges": [list(pair) for pair, _ in ordered],
        "specials": specials,
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_tokenizer(path: Path) -> tuple[dict[tuple[int, int], int], str | None, dict[str, int]]:
    """Recharge un tokenizer et renvoie (fusions, pattern, tokens spéciaux)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["format_version"] == FORMAT_VERSION, "Version de format inconnue"
    assert data["normalization"] == "NFC", "Normalisation non prise en charge"
    merges = {tuple(pair): 256 + k for k, pair in enumerate(data["merges"])}
    return merges, data["pattern"], data["specials"]



    


