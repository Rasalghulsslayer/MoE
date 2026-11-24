#Tests de l'index sémantique : forme, normalisation, déterminisme, Eiffel top-1, invariant position==id, bornes de k.
import json

import numpy as np
import pytest

from rag.embedding import EMBED_DIM, embed
from rag.index import build_index, search

MINI_BIBLIOTHEQUE = [
    "La tour Eiffel se trouve à Paris, sur le champ de Mars.",
    "Le BIOS est le premier code exécuté au démarrage d'un ordinateur.",
    "La photosynthèse transforme la lumière en énergie chimique chez les plantes.",
    "Montpellier est une ville du sud de la France, près de la mer Méditerranée.",
    "Python est un langage de programmation à typage dynamique.",
]


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


def test_contrat_de_forme():                     # A
    vecteurs = embed(["un passage", "un autre", "encore un"])
    assert vecteurs.shape == (3, EMBED_DIM)
    assert vecteurs.dtype == np.float32


def test_normalisation():                        # B — la condition cosine = produit scalaire
    vecteurs = embed(["un passage", "un autre"])
    assert np.allclose(np.linalg.norm(vecteurs, axis=1), 1.0, atol=1e-4)


def test_determinisme():                         # C
    assert np.allclose(embed(["le chat dort"]), embed(["le chat dort"]))


def test_eiffel_top1(mini_index):                # D — la propriété sémantique, faite main
    _, index = mini_index
    resultats = search("Où se trouve la tour Eiffel ?", k=3, index_path=index)
    assert resultats[0][0] == 0                  # le passage id 0 parle de la tour Eiffel


def test_position_egale_id(mini_index):          # E — l'id retrouvé relit la bonne ligne
    library, index = mini_index
    resultats = search("la photosynthèse chez les plantes", k=5, index_path=index)
    lignes = {json.loads(l)["id"]: json.loads(l) for l in open(library, encoding="utf-8")}
    for passage_id, _ in resultats:
        assert passage_id in lignes               # chaque id rendu désigne une vraie ligne
    assert resultats[0][0] == 2                   # le passage photosynthèse est le plus proche


def test_bornes_de_k(mini_index):                # F
    _, index = mini_index
    assert len(search("une question", k=2, index_path=index)) == 2
    assert len(search("une question", k=10, index_path=index)) == 5    # min(k, n), sans crash
    scores = [s for _, s in search("une question", k=5, index_path=index)]
    assert scores == sorted(scores, reverse=True)                     # triés décroissants