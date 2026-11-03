#Essayons notre tokenizer sur des données réelles
from pathlib import Path
from tokenizer.tokenizer import get_stats, merge, train, build_vocab, encode, decode, train_reg
from tokenizer.tries.try_regex import GPT4_PATTERN
from tokenizer.the_regex import REGEX_PATTERN
import time
import json
import random

TRAIN_SPLITS_PATH_FR = Path("data/splits/train_fr.jsonl")
VAL_SPLITS_PATH_FR = Path("data/splits/val_fr.jsonl")
TRAIN_SPLITS_PATH_CODE = Path("data/splits/train_code.jsonl")
VAL_SPLITS_PATH_CODE = Path("data/splits/val_code.jsonl")

def load_sample(path, source: str | None, max_chars):
    out = ""
    with open(path) as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            doc = json.loads(line)
            if source is None and len(out) < max_chars: 
                out += doc["text"] + "\n"
            elif doc["source"] == source and len(out) < max_chars:
                out += doc["text"] + "\n"
            else:
                continue
    return out

if __name__ == "__main__":
    print("Chargement des échantillons...")
    train_sample = load_sample(TRAIN_SPLITS_PATH_FR, "Wikipedia", 50000)
    val_sample_fr = load_sample(TRAIN_SPLITS_PATH_FR, None, 50000)
    val_sample_code = load_sample(TRAIN_SPLITS_PATH_CODE, None, 50000)
    print("Chargement terminé !")
    print("\n")


    print("=== Début de l'entraînement ===")
    #Temps d'entraînement pour la version classique (sans regex)
    print("--- Version classique (sans regex) ---")
    t_1 = time.perf_counter()
    merges_classic = train(train_sample, 512)
    print(len(merges_classic))
    t_2 = time.perf_counter()
    print(f"Temps de formation : {t_2 - t_1} secondes")

    #Temps d'entraînement pour la version regex avec GTP4_PATTERN
    print("--- Version regex avec GPT4_PATTERN ---")
    t_1 = time.perf_counter()
    merges_gpt = train_reg(train_sample, 512, GPT4_PATTERN)
    print(len(merges_gpt))
    t_2 = time.perf_counter()
    print(f"Temps de formation : {t_2 - t_1} secondes")

    #Temps d'entraînement pour la version regex avec REGEX_PATTERN
    print("--- Version regex avec REGEX_PATTERN ---")
    t_1 = time.perf_counter()
    merges_regex = train_reg(train_sample, 512, REGEX_PATTERN)
    print(len(merges_regex))
    t_2 = time.perf_counter()
    print(f"Temps de formation : {t_2 - t_1} secondes")

    print("\n")

    print("Encodage des échantillons...")
    # Encodage des échantillons avec les merges appris pour la version classique
    classic_encoded_train_sample = encode(train_sample, merges_classic)
    classic_encoded_val_sample_fr = encode(val_sample_fr, merges_classic)
    classic_encoded_val_sample_code = encode(val_sample_code, merges_classic)

    # Encodage des échantillons avec les merges appris pour la version gpt
    gpt_encoded_train_sample = encode(train_sample, merges_gpt)
    gpt_encoded_val_sample_fr = encode(val_sample_fr, merges_gpt)
    gpt_encoded_val_sample_code = encode(val_sample_code, merges_gpt)

    # Encodage des échantillons avec les merges appris pour la version regex
    regex_encoded_train_sample = encode(train_sample, merges_regex)
    regex_encoded_val_sample_fr = encode(val_sample_fr, merges_regex)
    regex_encoded_val_sample_code = encode(val_sample_code, merges_regex)
    print("Encodage terminé !")
    print("\n")


    print("=== Comparaison d'octets avant et après encodage ===")
    #Comparaison d'octets avant et après encodage: 
    print("--- Version classique (sans regex) ---")

    print("-train-")
    n_1 = len(train_sample.encode('utf-8'))
    print(f"Taille avant compression : {n_1}")
    n_2 = len(classic_encoded_train_sample)
    print(f"Taille après compression : {n_2}")
    r = n_1 / n_2
    print(f"Ratio : {r}")

    print("-val_fr-")
    n_1 = len(val_sample_fr.encode('utf-8'))
    print(f"Taille avant compression : {n_1}")
    n_2 = len(classic_encoded_val_sample_fr)
    print(f"Taille après compression : {n_2}")
    r = n_1 / n_2
    print(f"Ratio : {r}")

    print("-val_code-")
    n_1 = len(val_sample_code.encode('utf-8'))
    print(f"Taille avant compression : {n_1}")
    n_2 = len(classic_encoded_val_sample_code)
    print(f"Taille après compression : {n_2}")
    r = n_1 / n_2
    print(f"Ratio : {r}")
    print("\n")


    print("--- Version regex avec GPT4_PATTERN ---")

    print("-train-")
    n_1 = len(train_sample.encode('utf-8'))
    print(f"Taille avant compression : {n_1}")
    n_2 = len(gpt_encoded_train_sample)
    print(f"Taille après compression : {n_2}")
    r = n_1 / n_2
    print(f"Ratio : {r}")

    print("-val_fr-")
    n_1 = len(val_sample_fr.encode('utf-8'))
    print(f"Taille avant compression : {n_1}")
    n_2 = len(gpt_encoded_val_sample_fr)
    print(f"Taille après compression : {n_2}")
    r = n_1 / n_2
    print(f"Ratio : {r}")

    print("-val_code-")
    n_1 = len(val_sample_code.encode('utf-8'))
    print(f"Taille avant compression : {n_1}")
    n_2 = len(gpt_encoded_val_sample_code)
    print(f"Taille après compression : {n_2}")
    r = n_1 / n_2
    print(f"Ratio : {r}")
    print("\n")


    print("--- Version regex avec REGEX_PATTERN ---")
    
    print("-train-")
    n_1 = len(train_sample.encode('utf-8'))
    print(f"Taille avant compression : {n_1}")
    n_2 = len(regex_encoded_train_sample)
    print(f"Taille après compression : {n_2}")
    r = n_1 / n_2
    print(f"Ratio : {r}")

    print("-val_fr-")
    n_1 = len(val_sample_fr.encode('utf-8'))
    print(f"Taille avant compression : {n_1}")
    n_2 = len(regex_encoded_val_sample_fr)
    print(f"Taille après compression : {n_2}")
    r = n_1 / n_2
    print(f"Ratio : {r}")

    print("-val_code-")
    n_1 = len(val_sample_code.encode('utf-8'))
    print(f"Taille avant compression : {n_1}")
    n_2 = len(regex_encoded_val_sample_code)
    print(f"Taille après compression : {n_2}")
    r = n_1 / n_2
    print(f"Ratio : {r}")
    print("\n")

    vocab = build_vocab(merges_regex)
    #Affichage des nouveaux tokens:
    for i in range (256, 511):
        token = vocab[i].decode('utf-8', errors="replace")
        print(f"{i} : {token!r}")





