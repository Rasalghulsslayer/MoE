import torch
from model.transformer import Transformer
from model.config import DEV

def filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Filtre les logits pour ne garder que les k meilleurs."""
    top_k = min(top_k, logits.shape[-1])
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]   # forme (B, 1) : le k-ième score
    return logits.masked_fill(logits < threshold, float("-inf"))

def filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Filtre les logits pour que seul le plus petit ensemble atteignant p survive, le reste est set à -inf."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative - probs >= top_p
    sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
    return torch.full_like(logits, float('-inf')).scatter(-1, sorted_idx, sorted_logits)

def sample_next(logits: torch.Tensor, temperature: float, top_k = None, top_p = None) -> torch.Tensor:
    """Détermination du prochain token."""
    if temperature == 0: #Greedy
        return torch.argmax(logits, dim=-1, keepdim=True)
    else:
        logits = logits / temperature
    if top_k is not None:
        logits = filter_top_k(logits, top_k)
    if top_p is not None:
        logits = filter_top_p(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

@torch.no_grad()
def generate(model: Transformer, idx: torch.Tensor, max_new_tokens: int, temperature: float, top_k=None, top_p=None, use_cache=False) -> torch.Tensor:
    
    B, T = idx.shape

    assert T + max_new_tokens <= DEV.context_length, "LA genearation dépasse la taille de contexte autorisée."

    model.eval()

    if use_cache == False:
        for _ in range (max_new_tokens):
            idx = torch.cat((idx, sample_next(model(idx)[0][:, -1, :], temperature, top_k, top_p)), dim=1)
    else:
        caches = model.new_caches(B, idx.device)
        logits, _ = model(idx, caches=caches)
        for _ in range (max_new_tokens):
            new_tok = sample_next(logits[:,-1,:], temperature, top_k, top_p)
            idx = torch.cat((idx, new_tok), dim=1)
            logits, _ = model(new_tok, caches=caches)
    return idx
