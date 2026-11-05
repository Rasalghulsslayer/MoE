from dataclasses import dataclass
from tokenizer.special_tokens import VOCAB_SIZE

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int= VOCAB_SIZE
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_dim: int = 2048
    context_length: int = 1024
    sliding_window: int = 512
    rope_theta: float = 10_000.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("Dimensions incorrectes : d_model % n_heads != 0")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("Dimensions incorrectes : n_heads % n_kv_heads != 0")

    @property 
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

def count_params(cfg: ModelConfig) -> int:
    embedding_params = cfg.vocab_size * cfg.d_model
    transformer_params = cfg.d_model + (cfg.n_heads * cfg.head_dim) * cfg.d_model + 2 * ((cfg.n_kv_heads * cfg.head_dim) * cfg.d_model) + (cfg.n_heads * cfg.head_dim) * cfg.d_model + cfg.d_model + 2 * (cfg.d_model * cfg.ffn_dim) + cfg.ffn_dim * cfg.d_model
    if cfg.tie_embeddings:
        output_layer_params = cfg.d_model
    else :
        output_layer_params = cfg.d_model + embedding_params
    n_params = embedding_params + cfg.n_layers * transformer_params + output_layer_params
    return n_params


RUN = ModelConfig()

DEV = ModelConfig(
    d_model = 256,
    n_layers = 4,
    n_heads = 4,
    n_kv_heads = 2, #GQA, si n_kv_heads = 1 : MQA
    ffn_dim = 704,
    )