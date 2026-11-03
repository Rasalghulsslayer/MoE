#Testons les durées d'entraînement pour différentes tailles de données.
from pathlib import Path
from tokenizer.tokenizer import get_stats, merge, train, build_vocab, encode, decode, train_reg
from tokenizer.tries.try_regex import GPT4_PATTERN
from tokenizer.the_regex import REGEX_PATTERN
from utils.runtime import mem_mo
import time
import json
import random

TRAIN_SPLITS_PATH_FR = Path("data/splits/train_fr.jsonl")
TRAIN_SPLITS_PATH_CODE = Path("data/splits/train_code.jsonl")
ÉCHANTILLONS = [20000000] #[50000, 1000000, 5000000, 20000000] 
VOCAB_SIZE = [8192] #[512, 1024, 2048, 8192]

def load_train_sample(source: str | None, max_chars):
    out = ""
    with open(TRAIN_SPLITS_PATH_FR) as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            doc = json.loads(line)
            if source is None and len(out) < int(max_chars * 0.8): 
                out += doc["text"] + "\n"
            elif doc["source"] == source and len(out) < int(max_chars * 0.8):
                out += doc["text"] + "\n"
            else:
                continue
    with open(TRAIN_SPLITS_PATH_CODE) as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in lines:
                doc = json.loads(line)
                if source is None and len(out) < int(max_chars * 0.2): 
                    out += doc["text"] + "\n"
                elif doc["source"] == source and len(out) < int(max_chars * 0.2):
                    out += doc["text"] + "\n"
                else:
                    continue
    return out

if __name__ == "__main__":

    for i, (échantillon, vocab_size) in enumerate(zip(ÉCHANTILLONS, VOCAB_SIZE)):
        print(f"=== Passe {i + 1} / {len(ÉCHANTILLONS)} ===")

        print("Chargement des échantillons...")
        train_sample = load_train_sample(None, échantillon)
        print("Chargement terminé !")
        print("\n")


        print("=== Début de l'entraînement ===")
        #Temps d'entraînement pour la version classique (sans regex)
        print("Entrainement en cours...")
        t_1 = time.perf_counter()
        merges_regex = train_reg(train_sample, vocab_size)
        print(len(merges_regex))
        t_2 = time.perf_counter()
        print("Entrainement terminé !")
        print(f"Temps de formation : {t_2 - t_1} secondes")
        print(f"Pic mémoire : {mem_mo():.0f} Mo")
        print("\n")