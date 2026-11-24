import pytest

from rag.chunking import make_chunks
from rag.corpus import build_split

chunk_size = 256
overlap = 32
min_length = 64
VOCAB_TEST = {i: f"x{i} ".encode() for i in range (1500)}

def test