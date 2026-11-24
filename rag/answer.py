import json
import torch
import functools
from pathlib import Path
from rag.index import LIBRARY, INDEX, search
from tokenizer.tokenizer import encode, load_tokenizer, decode, build_vocab
from training.checkpoint import load_checkpoint
from model.generate import generate
from utils.runtime import get_device
from model.transformer import Transformer
from model.config import DEV 

BUDGET = 512
MAX_NEW_TOKENS = 40
CHECKPOINT = "latest.pt"
TOKENIZER_PATH = Path("artifacts/tokenizer/fr_code_32k.json")

@functools.cache
def load_model():
    data = load_checkpoint("latest.pt")
    model = Transformer(data["config"])
    model.load_state_dict(data["model"])
    model.to(get_device())
    model.eval()
    return model, data["config"]

@functools.cache
def load_tokenizer_cache():
    merges, pattern, specials = load_tokenizer(TOKENIZER_PATH)
    return merges, pattern, build_vocab(merges, specials=specials)


def get_passages(ids: list[int], library_path: Path = LIBRARY) -> list[str]:
    dict = {}
    with open(library_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            dict[doc["id"]] = doc["text"]
    passages = [dict[i] for i in ids]
    return passages

def build_prompt(question: str, passages: list[str], max_tokens: int = BUDGET) -> str:
    prompt = "Context: \n"
    for i in range (len(passages)):
        prompt += f"{i}: {passages[i]}\n" 
    prompt += "\n"
    prompt += f"Question : {question}\n"
    prompt += "\n"
    prompt += "Réponse :"
    return prompt[-max_tokens:]

def answer(question: str, k: int = 3, library_path=LIBRARY, index_path=INDEX, max_new_tokens: int = MAX_NEW_TOKENS) -> str:

    model, cfg = load_model
    merges, pattern, vocab = load_tokenizer_cache()

    search_results = search(question, k, index_path)
    ids =[]
    for id, _ in search_results:
        ids.append(id)
    passages = (get_passages(ids, library_path))
    prompt = build_prompt(question, passages, max_tokens=BUDGET)
    tkzed_prompt = encode(prompt, merges, pattern)
    assert len(tkzed_prompt) + max_new_tokens <= cfg.context_length, f"prompt {len(tkzed_prompt)} tokens + {max_new_tokens} générés > fenêtre {cfg.context_length} : coupe BUDGET ou k"
    
    device = next(model.parameters()).device
    entry = torch.tensor([tkzed_prompt], device=device)
    out = generate(model, entry, max_new_tokens, temperature=0.0)
    return decode(out[0].tolist(), vocab).strip()


if __name__ == "__main__":

    question = input("Posez votre question (la réponse utilisera la RAG):\n").strip()
    print(f"Top-3 meilleurs passages associés à la question:\n")
    search_results = search(question, k=3)
    ids =[]
    for id, _ in search_results:
        ids.append(id)
    passages = (get_passages(ids))
    for i in range(len(passages)):
        print(f"{i}. passage: {passages[i]} | score : {search_results[i][1]}\n")
    print("\n")
    print(f"Prompt final : \n {build_prompt(question, passages)} {answer(question)}" )
    