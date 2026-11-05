#Estimation des paramètres du modèle avant config

#Embedding
vocab_size = 32000
d_model = 768
embedding_dim = vocab_size * d_model == 24576000

#Transformer (x?)
head_dim = 64
n_heads = d_model // head_dim == 12
Q_dim = d_model * (head_dim * n_heads) == 589824
n_kv_heads = 4
K_dim = V_dim = d_model * (n_kv_heads * head_dim) == 196608
O_dim = (n_heads * head_dim) * d_model == 589824
ffn_dim = 2048
FFN_gate_dim = FFN_up_dim = d_model * ffn_dim == 1572864
FFN_down_dim = ffn_dim * d_model == 1572864
RMS_norm = d_model == 768 #(x2)
transformers_dim = RMS_norm + Q_dim + K_dim + V_dim + O_dim + RMS_norm + FFN_gate_dim + FFN_up_dim + FFN_down_dim == 6292992

#Output
RMS_norm_final = d_model

#Total
n_params = embedding_dim + "n_layers" * transformers_dim + RMS_norm_final == 125000000
n_layers = (n_params - RMS_norm_final - embedding_dim) // transformers_dim  + 1 == 16

#Embedding final, à ne pas oublier
reverse_embedding_dim = d_model * vocab_size == 24576000