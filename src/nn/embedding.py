import torch
import torch.nn as nn


dist_fn = lambda x, y: torch.sum((x - y) ** 2, dim=-1)


class Quantization(nn.Module):
    def __init__(self, vocab_size, embedding_dim, beta=0.25):
        super().__init__()
        self.beta = beta
        self.codebook = nn.Embedding(vocab_size, embedding_dim)

    @property
    def vocab_size(self):
        return self.codebook.num_embeddings

    @property
    def embedding_dim(self):
        return self.codebook.embedding_dim

    def forward(self, z, pos=None):
        idx = torch.vmap(lambda x, y: (x - y).abs().sum(-1), in_dims=(0, None)) (
            z, self.codebook.weight.detach()
        ).argmin(-1)
        z_q = self.codebook(idx)

        dist = dist_fn(z.detach(), z_q)
        if self.training:
            dist = (dist + self.beta * dist_fn(z, z_q.detach())) / 2  # commitment loss
            z_q = z + (z_q - z).detach()  # straight-through estimation
        return z_q, idx, dist

    def reset_parameters(self):
        with torch.no_grad():
            self.codebook.weight.uniform_(-1/self.vocab_size, 1/self.vocab_size)
