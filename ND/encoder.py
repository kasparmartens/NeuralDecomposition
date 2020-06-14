import torch
import torch.nn as nn

from torch.nn.functional import softplus


class cEncoder(nn.Module):
    """
    Encoder module for CVAE,
    i.e. it maps (Y, c) to the approximate posterior q(z)=N(mu_z, sigma_z)
    """

    def __init__(self, z_dim, mapping):
        super().__init__()

        self.z_dim = z_dim

        # NN mapping from (Y, x) to z
        self.mapping = mapping

    def forward(self, Y, c):

        out = self.mapping(torch.cat([Y, c], dim=1))

        mu = out[:, 0:self.z_dim]
        sigma = 1e-6 + softplus(out[:, self.z_dim:(2 * self.z_dim)])

        return mu, sigma
