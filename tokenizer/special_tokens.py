#On présente ici les tokens spéciaux utiles à l'ingestion des modèles qui complèteront notre vocabulaire

VOCAB_SIZE = 32_000
N_SPECIALS = 64
SPECIALS_START = VOCAB_SIZE - N_SPECIALS  # 31 936 : premier id spécial
NUM_MERGES = SPECIALS_START - 256         # 31 680 fusions à apprendre

NAMED = ["<s>", "</s>", "<pad>", "[INST]", "[/INST]", "<img>"]
RESERVED = [f"<reserved_{k}>" for k in range(N_SPECIALS - len(NAMED))]
SPECIALS = {name: SPECIALS_START + i for i, name in enumerate(NAMED + RESERVED)}

BOS_ID = SPECIALS["<s>"]
EOS_ID = SPECIALS["</s>"]
PAD_ID = SPECIALS["<pad>"]
