"""
Based on
https://github.com/brohrer/sharpened_cosine_similarity_torch/blob/main/sharpened_cosine_similarity.py
and copy/pasted heavily from code
https://github.com/ZeWang95/scs_pytorch/blob/main/scs.py
from Ze Wang
https://twitter.com/ZeWang46564905/status/1488371679936057348?s=20&t=lB_T74PcwZmlJ1rrdu8tfQ
and code
https://github.com/oliver-batchelor/scs_cifar/blob/main/src/scs.py
from Oliver Batchelor
https://twitter.com/oliver_batch/status/1488695910875820037?s=20&t=QOnrCRpXpOuC0XHApi6Z7A
and the TensorFlow implementation
https://colab.research.google.com/drive/1Lo-P_lMbw3t2RTwpzy1p8h0uKjkCx-RB
and blog post
https://www.rpisoni.dev/posts/cossim-convolution/
from Raphael Pisoni
https://twitter.com/ml_4rtemi5
"""
import torch
from torch import nn
import torch.nn.functional as F
# from opt_einsum import contract as einsum


class SharpenedCosineSimilarity(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=1, stride=1,
                 padding=0, p_scale=10, q_scale=100, eps=1e-12):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.padding = int(padding)

        self.p_scale = p_scale
        self.q_scale = q_scale

        self.q = nn.Parameter(torch.empty(1))
        self.p = nn.Parameter(torch.empty(out_channels))
        self.weights = nn.Parameter(torch.empty(out_channels, in_channels,
                                                kernel_size * kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.q, 10)
        nn.init.constant_(self.p, 2 ** .5 * self.p_scale)
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x):
        x = unfold2d(x, kernel_size=self.kernel_size, stride=self.stride,
                     padding=self.padding).flatten(start_dim=-2)

        # After unfolded and reshaped, dimensions of the images x are
        # dim 0, n: batch size
        # dim 1, c: number of input channels
        # dim 2, h: number of rows in the image
        # dim 3, w: number of columns in the image
        # dim 4, l: kernel size, squared
        #
        # The dimensions of the weights w are
        # dim 0, v: number of output channels
        # dim 1, c: number of input channels
        # dim 2, l: kernel size, squared

        q = self.q / self.q_scale

        x_norm = norm(x, [1, 4], q, self.eps)
        w_norm = norm(self.weights, [1, 2], q, self.eps)

        # use contract from opt_einsum
        x = torch.einsum('nchwl,vcl->nvhw', x / x_norm, self.weights / w_norm)
        # x = einsum('nchwl,vcl->nvhw', x / x_norm, self.weights / w_norm)

        sign = torch.sign(x)

        x = torch.abs(x) + self.eps
        x = x.pow(torch.square(self.p / self.p_scale).view(1, -1, 1, 1))
        return sign * x


def norm(x, dims, scale, eps):
    square_sum = torch.sum(x.square(), dims, keepdim=True)
    return (square_sum + eps).sqrt() + scale.square()


def unfold2d(x, kernel_size:int, stride:int, padding:int):
    x = F.pad(x, [padding]*4)
    bs, in_c, h, w = x.size()
    ks = kernel_size
    strided_x = x.as_strided(
        (bs, in_c, (h - ks) // stride + 1, (w - ks) // stride + 1, ks, ks),
        (in_c * h * w, h * w, stride * w, stride, w, 1))
    return strided_x
