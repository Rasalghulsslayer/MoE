#Tests 3.4 : juge de paix du trieur. Ecrits par le mentor, certifies contre le push.
import json
import re
import pytest
from rag.eval import QUESTIONS, K_MAX, hit_at_k


def _biblio() -> dict[int, dict]:
    """La vraie bibliothèque, lue une fois pour les tests de conformité."""
    docs = {}
    with open("rag/library/library.jsonl", encoding="utf-8") as f:
        for ligne in f:
            d = json.loads(ligne)
            docs[d["id"]] = d
    return docs


def test_rang_1_hit_partout():
    assert hit_at_k([[42, 7, 9]], [42], 1) == 1.0

def test_rang_3_miss_1_hit_3():
    assert hit_at_k([[7, 9, 42]], [42], 1) == 0.0
    assert hit_at_k([[7, 9, 42]], [42], 3) == 1.0

def test_absent_manque_a_tous_les_k():
    assert hit_at_k([[1, 2, 3]], [99], 5) == 0.0

def test_liste_vide_zero_sans_crash():
    """Le contrat : liste vide -> 0.0, SANS crash."""
    assert hit_at_k([], [], 3) == 0.0

def test_borne_stricte_du_slice():
    # l'attendu en position k+1 ne compte pas : la porte du top-k est fermée à k
    assert hit_at_k([[1, 2, 3, 4]], [4], 3) == 0.0
    assert hit_at_k([[1, 2, 3, 4]], [4], 4) == 1.0

def test_fraction_de_questions():
    assert hit_at_k([[5, 1], [2, 3], [4, 9]], [1, 3, 8], 2) == pytest.approx(2 / 3)

def test_crie_si_longueurs_differentes():
    with pytest.raises(AssertionError):
        hit_at_k([[1]], [1, 2], 1)


REPONSES = {9397: "Vincenzo Malo", 9173: "attaquant", 9296: "Léningrad", 9492: "Alun Francis",
            9430: "Laura Dianti", 9270: "Gauja", 9340: "Gudsi", 8588: "Bob Hoskins",
            8556: "1,93 m", 9214: "İstanbul Üniversitesi", 9131: "Paul Verlaine",
            3305: "Aston Villa", 3699: "Palais de glace Bolshoï", 6043: "Pierre Moscovici",
            6326: "près de quarante blessés", 3714: "Stieg Larsson", 6941: "QNB Group",
            2625: "Jean-Pierre Door", 8589: "Clayton Lockett", 8553: "Claudia Alexander"}


def test_20_questions_conformes_a_la_bibliotheque():
    docs = _biblio()
    assert len(QUESTIONS) == 20
    ids = [i for _, i in QUESTIONS]
    assert len(set(ids)) == 20              # un attendu par question, pas de doublon
    for question, attendu in QUESTIONS:
        assert question.strip() and question.endswith("?")
        assert attendu in docs, f"id {attendu} inexistant"
        assert docs[attendu]["language"] == "fr"
    assert K_MAX >= 5                       # le musée mesure hit@5 : il faut 5 candidats

def test_la_reponse_est_dans_le_passage_attendu():
    """Blancs normalisés : une réponse peut être coupée par un saut de ligne."""
    docs = _biblio()
    norm = lambda t: re.sub(r"\s+", " ", t.lower())
    for _, attendu in QUESTIONS:
        assert norm(REPONSES[attendu]) in norm(docs[attendu]["text"]), f"réponse absente du passage {attendu}"