from abc import abstractmethod
from typing import Any, Literal, Optional, Callable
from functools import partial

import numpy as np
import torch
import torch.nn as nn
# import torch.multiprocessing as mp
from torch.nn.functional import (
    mse_loss,
    log_softmax,
    l1_loss,
    binary_cross_entropy_with_logits as logits_bce,
    cross_entropy,
)
from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Optimizer, Adam
from scipy.optimize import linear_sum_assignment as lsa
from .math import (
    gauss2standard_kl,
    mmd_idxs,
    min_mean_discrepancy,
    inv_multiquad_sum,
    huber_norm,
    l2_norm,
    l1_norm,
)


class UpdatableLoss:
# class UpdatableLoss(metaclass=ABCMeta):
    # @classmethod
    # def __subclasshook__(cls, subclass):
    #     return hasattr(subclass, "update_parameters")

    @abstractmethod
    def update_parameters(self, *args, **kwargs):
        raise NotImplementedError


class ReconstructionLoss(Loss):
    def __init__(
        self,
        loss: Literal["bce", "mse", "l1"] = "mse",
    ):
        super().__init__()

        if loss == "bce":
            loss_fn = logits_bce
        elif loss == "mse":
            loss_fn = mse_loss
        elif loss == "l1":
            loss_fn = l1_loss
        else:
            raise ValueError(f"Unrecognized loss function {loss}")

        self.loss_fn = loss_fn

    def forward(self, input, target):
        if isinstance(input, (tuple, list)):
            recons = input[0]
        else:
            recons = input

        return self.loss_fn(recons, target, reduction="sum") / target.size(0)


class GaussianKL(Loss, UpdatableLoss):
    """
    This class implements the Variational Autoencoder loss with Multivariate
    Gaussian latent variables. With defualt parameters it is the one described
    in "Autoencoding Variational Bayes", Kingma & Welling (2014)
    [https://arxiv.org/abs/1312.6114].

    When $\beta>1$ this is the the loss described in $\beta$-VAE: Learning
    Basic Visual Concepts with a Constrained Variational Framework",
    Higgins et al., (2017) [https://openreview.net/forum?id=Sy2fzU9gl]
    """
    def __init__(self, beta=1.0, beta_schedule=None):
        super().__init__()
        self.beta = beta
        self.beta_schedule = beta_schedule
        self.anneal = 1.0

    def forward(self, z_sample, z_params):
        mu, logvar = z_params

        kl_div = gauss2standard_kl(mu, logvar).sum()
        kl_div /= z_sample.size(0)
        return self.anneal * self.beta * kl_div

    def update_parameters(self, step):
        if self.beta_schedule is not None:
            steps, schedule_type, min_anneal = self.beta_schedule
            delta = 1 / steps

            if schedule_type == 'anneal':
                self.anneal = max(1.0 - step * delta, min_anneal)
            elif schedule_type == 'increase':
                self.anneal = min(min_anneal + delta * step, 1.0)


class WassersteinAdversarial(Loss, UpdatableLoss):
    """
    Class that implements the adversarial version of the Wasserstein loss
    as found in "Wasserstein Autoencoders" Tolstikhin et al., 2019
    [https://arxiv.org/pdf/1711.01558.pdf].

    This version uses a trained discriminator to distinguish prior samples
    from posterior samples. The implementation is similar to FactorVAE,
    using a feedforward classifier. This model and the autoencoder are
    trained with conjugate gradient descent.
    """
    def __init__(self,
         lambda1=10.0,
         lambda2=0.0,
         prior_var=1.0,
         lmbda_schedule=None,
         discriminator: Optional[nn.Module] = None,
         optimizer: Optional[Callable[[Any], Optimizer]] = None
     ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.prior_var = prior_var
        self.lmbda_schedule = lmbda_schedule
        self.anneal = 1.0

        # if disc_args is None:
        #     disc_args = [('linear', [1000]), ('relu',)] * 6

        if discriminator is None:
            discriminator =  nn.Sequential(
                nn.Linear(10, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 2),
            )
        if optimizer is None:
            optimizer = partial(Adam, lr=1e-4, betas=(0.5, 0.9))

        self.disc = discriminator
        self.optim = optimizer(discriminator.parameters())
        self._batch_samples = None

    @property
    def disc_device(self):
        return next(self.disc.parameters()).device

    def train(self, mode=True):
        self.disc.train(mode)
        for p in self.disc.parameters():
            p.requires_grad = mode

    def eval(self):
        self.train(False)

    def _set_device(self, input_device):
        if self.disc_device is None or (self.disc_device != input_device):
            self.disc.to(device=input_device)

    def forward(self, z, z_params):
        # Hack to set the device
        self._set_device(z.device)
        self.eval()

        self._batch_samples = z.detach()

        log_z_ratio = self.disc(z)
        adv_term = (log_z_ratio[:, 0] - log_z_ratio[:, 1]).mean()

        if self.lambda2 != 0.0:
            logvar_reg = self.lambda2 * z_params[1].abs().sum() / z.size(0)
        else:
            logvar_reg = 0.0

        return self.anneal * self.lambda1 * adv_term + logvar_reg

    def update_parameters(self, step):
        # update anneal value
        if self.lmbda_schedule is not None:
            steps, min_anneal = self.lmbda_schedule
            delta = 1 / steps
            self.anneal = max(1.0 - step * delta, min_anneal)

        if self._batch_samples is None:
            return

        # Train discriminator
        self.train()
        self.optim.zero_grad()

        z, self._batch_samples = self._batch_samples, None

        z_prior = self.prior_var * torch.randn_like(z)

        log_ratio_z = self.disc(z)
        log_ratio_z_prior = self.disc(z_prior)

        ones = z_prior.new_ones(z_prior.size(0), dtype=torch.long)
        zeros = torch.zeros_like(ones)

        disc_loss = 0.5 * (cross_entropy(log_ratio_z, zeros) +
                           cross_entropy(log_ratio_z_prior, ones))

        disc_loss.backward()
        self.optim.step()


class WassersteinMMD(Loss, UpdatableLoss):
    """
    Class that implements the Minimum Mean Discrepancy term in the latent space
    as found in "Wasserstein Autoencoders", Tolstikhin et al., (2019)
    [https://arxiv.org/pdf/1711.01558.pdf], with the modifications proposed in
    "Learning disentangled representations with Wasserstein Autoencoders"
    Rubenstein et al., 2018 [https://openreview.net/pdf?id=Hy79-UJPM].

    Unlike the adversarial version, this one relies on kernels to determine the
    distance between the distributions. While we allow any kernel, the default
    is the sum of inverse multiquadratics which has heavier tails than RBF. We
    also add an L1 penalty on the log-variance to prevent the encoders from
    becoming deterministic.
    """
    def __init__(
        self,
        lambda1=10,
        lambda2=1.0,
        prior_type='norm',
        prior_var=1.0,
        kernel=None,
        lambda_schedule=None
    ):

        super().__init__()

        if kernel is None:
            kernel = partial(
                inv_multiquad_sum,
                base_scale=10.0,
                scales=torch.tensor([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
            )

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.prior_type = prior_type
        self.prior_var = prior_var
        self.lambda_schedule = lambda_schedule
        self.kernel = kernel
        self.anneal = 1.0
        # Save the indices of the combinations for reuse
        self._idxs = None

    def forward(self, z, z_params):
        if self.prior_type == 'norm':
            z_prior = self.prior_var * torch.randn_like(z)
        elif self.prior_type == 'unif':
            z_prior = self.prior_var * torch.rand_like(z) - 0.5
        else:
            raise ValueError('Unrecognized prior {}'.format(self.prior_type))

        if self._idxs is None or len(self._idxs[1]) != z.size(0) ** 2:
            self._idxs = mmd_idxs(z.size(0))

        adv_term = min_mean_discrepancy(z, z_prior, self.kernel, self._idxs)

        # L1 regularization of log-variance
        if self.lambda2 != 0.0:
            logvar_reg = self.lambda2 * z_params[1].abs().sum() / z.size(0)
        else:
            logvar_reg = 0.0

        return self.anneal * self.lambda1 * adv_term + logvar_reg

    def update_parameters(self, step):
        # update anneal value
        if self.lambda_schedule is not None:
            steps, min_anneal = self.lambda_schedule
            delta = 1 / steps
            self.anneal = max(1.0 - step * delta, min_anneal)


class ImageTokenLoss(Loss):
    def __init__(self):
        super().__init__(reduction='batchmean')

    def forward(self, inputs, targets):
        return -(targets * log_softmax(inputs, dim=-1)).sum() / len(inputs)


class HungarianAssignmentLoss(Loss):
    def __init__(self, loss='huber'):
        super().__init__(reduction='batchmean')
        if loss == 'huber':
            loss = huber_norm
        elif loss == 'l2':
            loss = l2_norm
        elif loss == 'l1':
            loss = l1_norm
        self.loss = loss

    def forward(self, inputs, targets):
        B = len(inputs)

        targets = targets.unsqueeze(1).expand(-1, inputs.size(1), -1, -1)
        inputs = inputs.unsqueeze(2).expand(-1, -1, targets.size(2), -1)

        pairwise_cost = self.loss(inputs , targets)

        # with mp.Pool(10) as p:
        #     assignment = p.map(lsa, pairwise_cost.detach().tolist())
        assignment = map(lsa, pairwise_cost.detach().tolist())

        idx_input, idx_targets = tuple(zip(*assignment))
        idx_input = np.array(idx_input)
        idx_targets = np.array(idx_targets)

        batch_idx = torch.arange(B).unsqueeze_(-1)

        return pairwise_cost[batch_idx, idx_input, idx_targets].sum() / B
