#Mission 3.4 : éval du trieur. Retrieval SEUL : la génération se mesure au musée
#(rag.answer, avant/après run long). Aucun score vu avant le gel de QUESTIONS.
from rag.index import search, INDEX   # le seul import : rien de torch ici

QUESTIONS: list[tuple[str, int]] = [
    ("Qui a été le maître du peintre Antonio Maria Vassallo ?", 9397),            # Vincenzo Malo
    ("À quel poste jouait le footballeur international Agu Casmir ?", 9173),     # attaquant
    ("Dans quelle ville est né le danseur Andrian Fadeïev ?", 9296),              # Léningrad
    ("Qui dirigeait l'Ulster Orchestra entre 1966 et 1976 ?", 9492),             # Alun Francis
    ("Qui était la mère d'Alphonse d'Este, fils du duc de Ferrare ?", 9430),     # Laura Dianti
    ("Dans quel cours d'eau le joueur d'échecs Alvis Vītoliņš a-t-il mis fin à ses jours ?", 9270),  # Gauja
    ("Sous quel nom de plume signait le poète Abbasgoulou agha Bakikhanov ?", 9340),  # Gudsi
    ("Quel acteur britannique est mort le 29 avril 2014 ?", 8588),               # Bob Hoskins
    ("Quelle est la taille du volleyeur argentin Guillermo Martínez ?", 8556),  # 1,93 m
    ("Pour quelle équipe turque a joué la basketteuse nigériane Aisha Balarabe en 2017-2018 ?", 9214),  # İstanbul Üniversitesi SK
    ("Quel poète le médecin Achille Souques a-t-il soigné durant ses études ?", 9131),  # Paul Verlaine
    ("Quel club a remporté la finale de la Coupe d'Europe des clubs champions en 1982 ?", 3305),  # Aston Villa
    ("Dans quelle patinoire les matchs de hockey du Canada se sont-ils joués aux JO de Sotchi ?", 3699),  # Palais de glace Bolshoï
    ("Quel ministre a relevé le plafond du PEA à 150 000 euros ?", 6043),        # Pierre Moscovici
    ("Combien de blessés a fait l'explosion d'une grenade à Antananarivo le jour de l'investiture présidentielle ?", 6326),  # près de quarante
    ("Quel auteur est décédé quelques jours après avoir déposé les manuscrits de sa trilogie chez son éditeur ?", 3714),  # Stieg Larsson
    ("Quelle banque a annoncé des résultats records pour les six premiers mois de 2013 ?", 6941),  # QNB Group
    ("Quel élu devait présenter l'amendement n° 83 sur l'assurance maladie à la séance ?", 2625),  # Jean-Pierre Door
    ("Quel condamné a été exécuté le 29 avril 2014 ?", 8589),                    # Clayton Lockett
    ("Quelle planétologue américaine est née le 30 mai 1959 ?", 8553),           # Claudia Alexander
]

K_MAX = 5   # UN seul search par question ; hit@1, hit@3 et hit@5 en découlent


def hit_at_k(rangs: list[list[int]], attendus: list[int], k: int) -> float:
    """Fraction de questions dont le passage attendu est dans le top-k.

    Entre : rangs[i] = les ids rendus par search pour la question i, classés
    du meilleur au pire score (longueur K_MAX pour tout mesurer) ;
    attendus[i] = l'id du passage qui contient la réponse ; k = taille du top.


    Rend : nombre de questions où attendus[i] figure dans rangs[i][:k],
    divisé par le nombre de questions. Liste vide -> 0.0, sans crash.

    Pure : zéro search, zéro I/O, zéro bibliothèque. Testable par rangs forgés.
    Doit crier (assert) si len(rangs) != len(attendus).
    """
    score = 0.0
    n = len(rangs)
    assert n == len(attendus)
    for i in range (n):
        if attendus[i] in rangs[i][:k]:
            score += 1
    return score/n



if __name__ == "__main__":
    # Le musée, en trois temps :
    # 1) Pour chaque question : UN search(question, K_MAX) -> ids + scores.
    # 2) Verdict par question : affiche le top-3 (id, score) + où s'est classé
    #    l'attendu : « rang 1 », « rang 3 », ou « absent ».
    # 3) Bannière finale, ex : hit@1 0.60 (12/20) | hit@3 0.75 (15/20) | hit@5 0.80 (16/20)
    rangs = []
    attendus = []
    for question, attendu in QUESTIONS:
        ranks = []
        attendus.append(attendu)
        result_search = search(question, K_MAX, INDEX)
        for rank, score in result_search:
            ranks.append(rank)
        rangs.append(ranks)
    print(f"hit@1: {hit_at_k(rangs, attendus, 1)}")
    print(f"hit@3: {hit_at_k(rangs, attendus, 3)}")
    print(f"hit@5: {hit_at_k(rangs, attendus, 5)}")
    