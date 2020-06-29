import torch
import numpy as np

import torch.nn as nn
from torch.autograd import Variable

from torch.nn.functional import softplus

from .helpers import expand_grid, approximate_KLqp, rsample_RelaxedBernoulli

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs

class Decoder(nn.Module):

    def __init__(self, output_dim,
                 grid_z, grid_c, grid_cz,
                 mapping_z=None, mapping_c=None, mapping_cz=None,
                 has_feature_level_sparsity=True,
                 penalty_type="fixed", lambda0=1.0,
                 likelihood="Gaussian",
                 p1=0.2, p2=0.2, p3=0.2, device="cpu"):
        """
        NN mapping with constraints to be used as the decoder in a CVAE. Performs Neural Decomposition.
        :param output_dim: data dimensionality
        :param grid_z: grid for quadrature (estimation of integral for f(z))
        :param grid_c: grid for quadrature (estimation of integral for f(c))
        :param grid_cz: grid for quadrature (estimation of integral for f(z, c))
        :param mapping_z: neural net mapping z to data
        :param mapping_c: neural net mapping c to data
        :param mapping_cz: neural net mapping (z, c) to data
        :param has_feature_level_sparsity: whether to use (Relaxed) Bernoulli feature-level sparsity
        :param penalty_type: which penalty to apply
        :param lambda0: initialisation for both fixed penalty $c$ as well as $lambda$ values
        :param likelihood: Gaussian or Bernoulli
        :param p1: Bernoulli prior for sparsity on mapping_z
        :param p2: Bernoulli prior for sparsity on mapping_c
        :param p3: Bernoulli prior for sparsity on mapping_zc
        :param device: cpu or cuda
        """
        super().__init__()

        self.output_dim = output_dim
        self.likelihood = likelihood
        self.has_feature_level_sparsity = has_feature_level_sparsity
        self.penalty_type = penalty_type

        self.grid_z = grid_z.to(device)
        self.grid_c = grid_c.to(device)
        self.grid_cz = grid_cz.to(device)

        self.n_grid_z = grid_z.shape[0]
        self.n_grid_c = grid_c.shape[0]
        self.n_grid_cz = grid_cz.shape[0]

        # input -> output
        self.mapping_z = mapping_z
        self.mapping_c = mapping_c
        self.mapping_cz = mapping_cz

        if self.likelihood == "Gaussian":
            # feature-specific variances (for Gaussian likelihood)
            self.noise_sd = torch.nn.Parameter(-1.0 * torch.ones(1, output_dim))

        self.intercept = torch.nn.Parameter(torch.zeros(1, output_dim))

        self.Lambda_z = Variable(lambda0*torch.ones(1, output_dim, device=device), requires_grad=True)

        self.Lambda_c = Variable(lambda0*torch.ones(1, output_dim, device=device), requires_grad=True)

        self.Lambda_cz_1 = Variable(lambda0*torch.ones(self.n_grid_z, output_dim, device=device), requires_grad=True)

        self.Lambda_cz_2 = Variable(lambda0*torch.ones(self.n_grid_c, output_dim, device=device), requires_grad=True)

        self.lambda0 = lambda0

        self.device = device

        # RelaxedBernoulli
        self.temperature = 1.0 * torch.ones(1, device=device)

        if self.has_feature_level_sparsity:

            # for the prior RelaxedBernoulli(logits)
            self.logits_z = probs_to_logits(p1 * torch.ones(1, output_dim).to(device), is_binary=True)
            self.logits_c = probs_to_logits(p2 * torch.ones(1, output_dim).to(device), is_binary=True)
            self.logits_cz = probs_to_logits(p3 * torch.ones(1, output_dim).to(device), is_binary=True)

            # for the approx posterior
            self.qlogits_z = torch.nn.Parameter(3.0 * torch.ones(1, output_dim).to(device))
            self.qlogits_c = torch.nn.Parameter(3.0 * torch.ones(1, output_dim).to(device))
            self.qlogits_cz = torch.nn.Parameter(2.0 * torch.ones(1, output_dim).to(device))


    def forward_z(self, z):
        value = self.mapping_z(z)
        if self.has_feature_level_sparsity:
            w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_z)
            return w * value
        else:
            return value

    def forward_c(self, c):
        value = self.mapping_c(c)
        if self.has_feature_level_sparsity:
            w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_c)
            return w * value
        else:
            return value

    def forward_cz(self, z, c):
        return self.forward_cz_concat(torch.cat([z, c], dim=1))

    def forward_cz_concat(self, zc_concat):
        value = self.mapping_cz(zc_concat)
        if self.has_feature_level_sparsity:
            w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_cz)
            return w * value
        else:
            return value


    def forward(self, z, c):
        return self.intercept + self.forward_z(z) + self.forward_c(c) + self.forward_cz(z, c)

    def loglik(self, y_pred, y_obs):

        if self.likelihood == "Gaussian":
            sigma = 1e-6 + softplus(self.noise_sd)
            p_data = Normal(loc=y_pred, scale=sigma)
            loglik = p_data.log_prob(y_obs).sum()
        elif self.likelihood == "Bernoulli":
            p_data = Bernoulli(logits=y_pred)
            loglik = p_data.log_prob(y_obs).sum()
        else:
            raise NotImplementedError("Other likelihoods not implemented")

        return loglik

    def set_temperature(self, x):
        self.temperature = x * torch.ones(1, device=self.device)

    def calculate_integrals(self):

        # has shape [1, output_dim]
        int_z = self.forward_z(self.grid_z).mean(dim=0).reshape(1, self.output_dim)

        # has shape [1, output_dim]
        int_c = self.forward_c(self.grid_c).mean(dim=0).reshape(1, self.output_dim)

        m1 = self.n_grid_z
        m2 = self.n_grid_c

        out = self.forward_cz_concat(self.grid_cz)
        out = out.reshape(m1, m2, self.output_dim)

        # has shape [m1, output_dim]
        int_cz_dc = out.mean(dim=1)
        # has shape [m2, output_dim]
        int_cz_dz = out.mean(dim=0)

        return int_z, int_c, int_cz_dc, int_cz_dz

    def calculate_integrals_numpy(self):

        with torch.no_grad():

            # has shape [1, output_dim]
            int_z = self.forward_z(self.grid_z).mean(dim=0).reshape(1, self.output_dim).cpu().numpy()

            # has shape [1, output_dim]
            int_c = self.mapping_c(self.grid_c).mean(dim=0).reshape(1, self.output_dim).cpu().numpy()

            m1 = self.n_grid_z
            m2 = self.n_grid_c

            out = self.forward_cz_concat(self.grid_cz)
            out = out.reshape(m1, m2, self.output_dim)

            # has shape [m1, output_dim]
            int_cz_dc = out.mean(dim=1).cpu().numpy()
            # has shape [m2, output_dim]
            int_cz_dz = out.mean(dim=0).cpu().numpy()

            int_cz = np.vstack((int_cz_dc, int_cz_dz))

        return int_z, int_c, int_cz


    def calculate_penalty(self):
        int_z, int_c, int_cz_1, int_cz_2 = self.calculate_integrals()

        # penalty with fixed lambda0
        if self.penalty_type in ["fixed", "MDMM"]:
            penalty0 = self.lambda0 * (int_z.abs().mean() + int_c.abs().mean() + \
                                       int_cz_1.abs().mean() + int_cz_2.abs().mean())

        if self.penalty_type in ["BDMM", "MDMM"]:
            penalty_BDMM = (self.Lambda_z * int_z).mean() + (self.Lambda_c * int_c).mean() + \
                           (self.Lambda_cz_1 * int_cz_1).mean() + (self.Lambda_cz_2 * int_cz_2).mean()


        if self.penalty_type == "fixed":
            penalty = penalty0
        elif self.penalty_type == "BDMM":
            penalty = penalty_BDMM
        elif self.penalty_type == "MDMM":
            penalty = penalty_BDMM + penalty0
        else:
            raise ValueError("Unknown penalty type")

        return penalty, int_z, int_c, int_cz_1, int_cz_2

    def loss(self, y_pred, y_obs):

        penalty, int_z, int_c, int_cz_dc, int_cz_dz = self.calculate_penalty()

        total_loss = - self.loglik(y_pred, y_obs) + penalty

        if self.has_feature_level_sparsity:
            KL1 = approximate_KLqp(self.logits_z, self.qlogits_z)
            KL2 = approximate_KLqp(self.logits_c, self.qlogits_c)
            KL3 = approximate_KLqp(self.logits_cz, self.qlogits_cz)
            total_loss += 1.0 * (KL1 + KL2 + KL3)

        return total_loss, penalty, int_z, int_c, int_cz_dc, int_cz_dz

    def fraction_of_variance_explained(self, z, c, account_for_noise=False, divide_by_total_var=True):

        with torch.no_grad():
            # f_z effect
            f_z = self.forward_z(z)
            f_z_var = f_z.var(dim=0, keepdim=True)

            # f_c
            f_c = self.forward_c(c)
            f_c_var = f_c.var(dim=0, keepdim=True)

            # f_int
            f_int = self.forward_cz(z, c)
            f_int_var = f_int.var(dim=0, keepdim=True)

            # collect Var([f_z, f_c, f_int]) together
            # and divide by total variance
            f_all_var = torch.cat([f_z_var, f_c_var, f_int_var], dim=0)

            if divide_by_total_var:

                total_var = f_all_var.sum(dim=0, keepdim=True)

                if account_for_noise:
                    total_var += self.noise_sd.reshape(-1) ** 2

                f_all_var /= total_var

            return f_all_var.t()

    def get_feature_level_sparsity_probs(self):

        with torch.no_grad():
            # f_z effect
            w_z = torch.sigmoid(self.qlogits_z)
            w_c = torch.sigmoid(self.qlogits_c)
            w_cz = torch.sigmoid(self.qlogits_cz)

            return torch.cat([w_z, w_c, w_cz], dim=0).t()
