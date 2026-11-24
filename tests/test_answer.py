import pytest
import json
from test_index import MINI_BIBLIOTHEQUE
from rag.answer import get_passages, build_prompt, answer
from rag.index import build_index

@pytest.fixture
def mini_index(tmp_path):
    """Une bibliothèque de 5 passages forgés + son index, dans tmp_path (jamais sur la vraie)."""
    library = tmp_path / "library.jsonl"
    with open(library, "w", encoding="utf-8") as f:
        for i, text in enumerate(MINI_BIBLIOTHEQUE):
            f.write(json.dumps({"id": i, "doc_id": f"doc{i}", "source": "manuel",
                                "language": "fr", "n_tokens": len(text.split()), "text": text},
                               ensure_ascii=False) + "\n")
    index = tmp_path / "index.faiss"
    build_index(library, index)
    return library, index



def test_get_passages():
    ids = [2, 0]
    library, _ = mini_index
    assert get_passages(ids, library) == [MINI_BIBLIOTHEQUE[2], MINI_BIBLIOTHEQUE[0]]

def test_build_prompt():
    question = "Pourquoi les mouches n'ont elles pas de dents ?"
    prompt = build_prompt(question, [3,1])
    assert question in prompt
    assert MINI_BIBLIOTHEQUE[3] in prompt
    assert MINI_BIBLIOTHEQUE[1] in prompt
    assert prompt == build_prompt(question, [3,1])

def test_build_prompt_scale():
    question ="Pourquoi Pourquoi Pourquoi Pourquoi Pourquoi Pourquoi Pourquoi Pourquoi Pourquoi Pourquoi Pourquoi Pourquoi Pourquoi Pourquoi Pourquoi Pourquoi les mouches n'ont elles pas de dents ?"
    prompt = build_prompt(question, [1, 2, 3, 4], max_tokens=50)
    assert len(prompt) == 50

def test_answer():
    