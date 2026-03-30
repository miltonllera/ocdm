import torch
import torch.nn as nn


dist_fn = lambda x, y: torch.sum((x - y) ** 2, dim=-1)


class Quantization(nn.Module):
    """
    Standard quantization layer using a flat codebook which must learn all the possible patch types.
    """
    def __init__(self, vocab_size, embedding_dim, beta=1.0):
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
            dist = 0.5 * (dist + self.beta * dist_fn(z, z_q.detach()))  # commitment loss
            z_q = z + (z_q - z).detach()  # straight-through estimation
        return z_q, idx, dist

    def reset_parameters(self):
        with torch.no_grad():
            self.codebook.weight.uniform_(-1/self.vocab_size, 1/self.vocab_size)


class ResidualQuantization(nn.Module):
    """
    Residual quantization layer which decomposes a vector into a sum of residuals. Larger depths
    mean potentially better approximations.
    """
    def __init__(self, vocab_size, embedding_dim, depth=4, beta=1.0):
        super().__init__()
        self.beta = beta
        self.codebook = nn.Embedding(vocab_size, embedding_dim)
        self.depth = depth

    @property
    def vocab_size(self):
        return self.codebook.num_embeddings

    @property
    def embedding_dim(self):
        return self.codebook.embedding_dim

    def get_quantization(self, z):
        idx = torch.vmap(lambda x, y: (x - y).abs().sum(-1), in_dims=(0, None)) (
            z, self.codebook.weight.detach()
        ).argmin(-1)
        return self.codebook(idx), idx

    def forward(self, z: torch.Tensor, pos=None):
        residual, zq, idxs = z, z.new_zeros((1,)), []
        for _ in range(self.depth):
            rq, idx = self.get_quantization(residual)
            zq = zq + rq
            residual = residual - rq
            idxs.append(idx)

        dist = dist_fn(z.detach(), zq)
        if self.training:
            dist = 0.5 * (dist + self.beta * dist_fn(z, zq.detach()))  # commitment loss
            zq = z + (zq - z).detach()  # straight-through estimation

        return zq, torch.stack(idxs, dim=-1), dist

    def reset_parameters(self):
        with torch.no_grad():
            self.codebook.weight.uniform_(-1/self.vocab_size, 1/self.vocab_size)


class CombinatorialQuantization(nn.Module):
    """
    Quantization which partitions the encoding space into separate spaces with disjoint codebooks.
    These can be combined combinatorially to generate the feature at a particular location.
    """
    def __init__(self, vocab_size, embedding_dim, n_partitions=4, beta=1.0):
        super().__init__()
        self.beta = beta
        self.codebook = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim) for _ in range(n_partitions)
        ])

    @property
    def vocab_size(self):
        return self.codebook[0].num_embeddings

    @property
    def embedding_dim(self):
        return self.codebook[0].embedding_dim

    @property
    def n_partitions(self):
        return len(self.codebook)

    def get_quantization(self, z, i):
        codebook = self.codebook[i]
        idx = torch.vmap(
            lambda x, y: (x - y).abs().sum(-1), in_dims=(0, None)
        )(z, codebook).argmin(-1)
        return self.codebook(idx), idx

    def forward(self, z: torch.Tensor, pos=None):
        z_part, idxs, z_q = z.chunk(self.n_partitions, dim=-1), [], []
        for i, zp in enumerate(z_part):
            zp_q, idx = self.get_quantization(zp, i)
            z_q.append(zp_q)
            idxs.append(idx)

        z_q = torch.cat(z_q, dim=-1)
        idxs = torch.stack(idxs, dim=-1)
        dist = dist_fn(z.detach(), z_q)

        if self.training:
            dist = 0.5 * (dist + self.beta * dist_fn(z, z_q.detach()))  # commitment loss
            z_q = z + (z_q - z).detach()  # straight-through estimation

        return z_q, idxs, dist

    def reset_parameters(self):
        with torch.no_grad():
            for codebook in self.codebook:
                codebook.weight.uniform_(-1/ self.vocab_size, 1/ self.vocab_size)  # type: ignore
