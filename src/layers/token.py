import torch.nn as nn
import torch.nn.functional as F
from .spatial import PositionEmbedding1D


def to_onehot(token_probs):
    # one_hot preserves device information but is a stop-gradient operation
    index = token_probs.argmax(dim=-1)
    return F.one_hot(index, token_probs.shape[-1])


class TokenDict(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len):
        super().__init__()
        self.embedding_dict = nn.Embedding(vocab_size, embedding_dim)
        self.add_pos_emb = PositionEmbedding1D(max_seq_len, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    @property
    def vocab_size(self):
        return self.embedding_dict.num_embeddings

    @property
    def embedding_dim(self):
        return self.embedding_dict.embedding_dim

    def forward(self, index_weights, start_pos=0):
        index = index_weights.argmax(dim=-1)
        embeddings = self.embedding_dict(index)
        return self.dropout(self.add_pos_emb(embeddings, start_pos))

    def reset_parameters(self):
        self.embedding_dict.reset_parameters()
        self.add_pos_emb.reset_parameters()
