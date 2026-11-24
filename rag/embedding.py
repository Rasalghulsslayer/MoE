import functools

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384

@functools.cache
def load_encoder():
    return SentenceTransformer(EMBED_MODEL)

def embed(texts: list[str], encoder: SentenceTransformer | None = None, batch_size: int = 64) -> np.ndarray:
    if encoder is None:
        encoder = load_encoder()
    vectors = encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(vectors, dtype=np.float32)