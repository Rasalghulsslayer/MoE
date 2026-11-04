from tokenizer.tokenizer import get_stats, merge, train, build_vocab, encode, decode, encode_chunks, train_reference, train_reg, save_tokenizer, load_tokenizer
from tokenizer.the_regex import REGEX_PATTERN
import pytest
import unicodedata
from tokenizer.special_tokens import SPECIALS, SPECIALS_START, VOCAB_SIZE, BOS_ID, EOS_ID
from tokenizer.tokenizer import encode_document


def test_get_stats():
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_merge_overlap():
    assert merge([5, 5, 5], (5, 5), 9) == [9, 5]

def test_train_toy():
    merges = train("aaabdaaabac", 259)
    assert merges == {(97, 97): 256, (97, 98): 257, (256, 257): 258}


def test_build_vocab():
    vocab = build_vocab(train("aaabdaaabac", 259))
    assert vocab[258] == b"aaab"    

def test_train_short_text():
    assert train("ab", 300) == {(97, 98): 256}

TOY = train("aaabdaaabac", 259)


def test_encode_toy():
    assert encode("aaabdaaabac", TOY) == [258, 100, 258, 97, 99]


@pytest.mark.parametrize("text", ["", "a", "L'été à Paris 🌞", "e\u0301", "日本語",
                                  "def f(x):\n    return x"])
def test_roundtrip(text):
    assert decode(encode(text, TOY), build_vocab(TOY)) == text

@pytest.mark.parametrize("text", ["", "a", "L'été à Paris 🌞", "e\u0301", "日本語",
                                  "def f(x):\n    return x"])
def test_roundtrip_reg(text):
    assert decode(encode_chunks(text, TOY), build_vocab(TOY)) == unicodedata.normalize("NFC", text)

def test_decode_partial_char():
    assert decode([195], build_vocab(TOY)) == "\ufffd"

def test_encode_twice_same_result():
    merges = train("aaabdaaabac", 259)
    assert encode("aaabdaaabac", merges) == encode("aaabdaaabac", merges)

def test_encode_is_concatenation_of_chunks():
    merges = train("le chat le chien", 270)
    expected = encode("le", merges) + encode(" chat", merges)
    assert encode_chunks("le chat", merges) == expected


TEXT = ("Le chat de la voisine dort sur le canapé. Le chien de Paul aboie dans le jardin. "
        "Les enfants de l'école jouent dans la cour pendant que la maîtresse les surveille. ") * 5

def test_incremental_matches_reference():
    assert train_reg(TEXT, 400) == train_reference(TEXT, 400)

def test_specials_layout():
    assert len(SPECIALS) == 64
    assert min(SPECIALS.values()) == SPECIALS_START
    assert max(SPECIALS.values()) == VOCAB_SIZE - 1


def test_special_text_is_not_special_token():
    merges = train("Le chat dort. [INST] Bonjour [/INST] <s>", 300)
    ids = encode("[INST] <s>", merges)
    assert all(idx < SPECIALS_START for idx in ids)


def test_encode_document():
    merges = train("Le chat dort.", 270)
    vocab = build_vocab(merges, SPECIALS)
    ids = encode_document("Le chat dort.", merges)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert decode(ids, vocab) == "<s>Le chat dort.</s>"

def test_save_load_roundtrip(tmp_path):
    text = "Le chat de la voisine dort. Le chien de Paul aboie."
    merges = train(text, 300)
    path = tmp_path / "tok.json"
    save_tokenizer(path, merges, REGEX_PATTERN, SPECIALS, {"test": True})
    loaded_merges, loaded_pattern, loaded_specials = load_tokenizer(path)
    assert loaded_merges == merges
    assert loaded_pattern == REGEX_PATTERN
    assert loaded_specials == SPECIALS
    assert encode_chunks(text, loaded_merges, loaded_pattern) == encode_chunks(text, merges, REGEX_PATTERN)




