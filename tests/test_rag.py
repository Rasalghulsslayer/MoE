#Tests de la bibliothèque RAG : découpage, reconstruction (l'oracle reine), ids, cas limites.
import pytest

from rag.chunking import make_chunks
from tokenizer.tokenizer import decode

CS, OV, MN = 256, 32, 64
VOCAB_TEST = {i: f"x{i} ".encode() for i in range(1500)}   # vocabulaire factice : les propriétés n'en dépendent pas


def chunks_de(n, source="src", **kwargs):
    return make_chunks(list(range(n)), VOCAB_TEST, CS, OV, source, MN, **kwargs)


def reconstruit(chunks):
    """L'oracle reine : concat en retirant l'overlap de chaque voisin."""
    r = []
    for c in chunks:
        r.extend(c["token_ids"] if not r else c["token_ids"][OV:])
    return r


def test_1200_tokens_et_overlap_reel():
    chunks, stats = chunks_de(1200, id_start=100)
    assert stats == {"passages": 6, "tokens": 1200, "fusions": 0}
    assert [c["id"] for c in chunks] == [100, 101, 102, 103, 104, 105]
    assert [len(c["token_ids"]) for c in chunks] == [256, 256, 256, 256, 256, 80]
    assert reconstruit(chunks) == list(range(1200))              # A. reconstruction
    for a, b in zip(chunks, chunks[1:]):
        assert a["token_ids"][-OV:] == b["token_ids"][:OV]       # C. overlap réel
        assert a["text"] == decode(a["token_ids"], VOCAB_TEST)  # le texte passe par le vocab fourni


def test_1000_queue_courte():
    chunks, stats = chunks_de(1000)
    assert stats == {"passages": 5, "tokens": 1000, "fusions": 0}
    assert [len(c["token_ids"]) for c in chunks] == [256, 256, 256, 256, 104]
    assert reconstruit(chunks) == list(range(1000))
    for a, b in zip(chunks, chunks[1:]):
        assert a["token_ids"][-OV:] == b["token_ids"][:OV]


def test_queue_avalee():
    chunks, stats = chunks_de(1160)          # la queue (40 nouveaux tokens) est avalée
    assert stats == {"passages": 5, "tokens": 1160, "fusions": 1}
    assert [len(c["token_ids"]) for c in chunks] == [256, 256, 256, 256, 264]
    assert reconstruit(chunks) == list(range(1160))


def test_448_et_300():
    chunks, stats = chunks_de(448)
    assert stats == {"passages": 2, "tokens": 448, "fusions": 0}
    assert [len(c["token_ids"]) for c in chunks] == [256, 224]
    assert reconstruit(chunks) == list(range(448))
    chunks, stats = chunks_de(300)
    assert stats == {"passages": 2, "tokens": 300, "fusions": 0}
    assert [len(c["token_ids"]) for c in chunks] == [256, 76]
    assert reconstruit(chunks) == list(range(300))


def test_documents_courts():
    for n in (256, 200, 64):        # division exacte, court, plus court que min_len : UN passage, zéro fusion
        chunks, stats = chunks_de(n)
        assert chunks[0]["token_ids"] == list(range(n))
        assert stats == {"passages": 1, "tokens": n, "fusions": 0}


def test_document_vide():
    chunks, stats = make_chunks([], VOCAB_TEST, CS, OV, "src", MN)
    assert chunks == []
    print(stats)
    print({"passages": 0, "tokens": 0, "fusions": 0})
    assert stats == {"passages": 0, "tokens": 0, "fusions": 0}


def test_id_start_et_source():
    chunks, stats = chunks_de(700, source="Wikipedia-fr", id_start=9)
    assert [c["id"] for c in chunks] == [9, 10, 11]
    assert [len(c["token_ids"]) for c in chunks] == [256, 256, 252]
    assert all(c["source"] == "Wikipedia-fr" for c in chunks)
    assert stats == {"passages": 3, "tokens": 700, "fusions": 0}
    assert reconstruit(chunks) == list(range(700))


def test_garde_decoupage_impossible():
    for mauvais in (256, 257, -1):
        with pytest.raises(ValueError):
            make_chunks(list(range(100)), VOCAB_TEST, CS, mauvais, "src", MN)