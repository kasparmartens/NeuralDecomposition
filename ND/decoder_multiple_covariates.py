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

class Decoder_multiple_covariates(nn.Module):

    def __init__(self, output_dim, n_covariates,
                 grid_z, grid_c,
                 mapping_z, mappings_c, mappings_cz,
                 has_feature_level_sparsity=True,
                 penalty_type="fixed", lambda0=1.0,
                 likelihood="Gaussian",
                 p1=0.2, p2=0.2, p3=0.2, device="cpu"):
        """
        Decoder for multiple covariates (i.e. multivariate c)
        """
        super().__init__()

        self.output_dim = output_dim
        self.likelihood = likelihood
        self.has_feature_level_sparsity = has_feature_level_sparsity
        self.penalty_type = penalty_type
        self.n_covariates = n_covariates

        assert isinstance(grid_c, list), "grid_c must be a list"
        assert len(grid_c) == n_covariates

        self.grid_z = grid_z
        self.grid_c = grid_c
        self.grid_cz = [
            torch.cat(expand_grid(self.grid_z, self.grid_c[j]), dim=1) for j in range(self.n_covariates)
        ]

        self.grid_z = grid_z.to(device)
        self.grid_c = [c.to(device) for c in grid_c]
        self.grid_cz = [cz.to(device) for cz in self.grid_cz]

        self.n_grid_z = grid_z.shape[0]
        self.n_grid_c = [c.shape[0] for c in grid_c]
        self.n_grid_cz = [cz.shape[0] for cz in self.grid_cz]

        # input -> output
        self.mapping_z = mapping_z
        self.mappings_c = mappings_c
        self.mappings_cz = mappings_cz

        if self.likelihood == "Gaussian":
            # feature-specific variances (for Gaussian likelihood)
            self.noise_sd = torch.nn.Parameter(-1.0 * torch.ones(1, output_dim))

        self.intercept = torch.nn.Parameter(torch.zeros(1, output_dim))

        self.Lambda_z = Variable(torch.ones(1, output_dim, device=device), requires_grad=True)

        self.Lambda_c = [
            Variable(torch.ones(n_covariates, 1, output_dim, device=device), requires_grad=True) for _ in range(self.n_covariates)
        ]

        self.Lambda_cz_1 = [
            Variable(torch.ones(self.n_grid_z, output_dim, device=device), requires_grad=True) for _ in range(self.n_covariates)
        ]

        self.Lambda_cz_2 = [
            Variable(torch.ones(self.n_grid_c[j], output_dim, device=device), requires_grad=True) for j in range(self.n_covariates)
        ]

        self.lambda0 = lambda0

        self.device = device

        # RelaxedBernoulli
        self.temperature = 1.0 * torch.ones(1, device=device)

        if self.has_feature_level_sparsity:

            # for the prior RelaxedBernoulli(logits)
            self.logits_z = probs_to_logits(p1 * torch.ones(1, output_dim).to(device), is_binary=True)
            self.logits_c = probs_to_logits(p2 * torch.ones(n_covariates, output_dim).to(device), is_binary=True)
            self.logits_cz = probs_to_logits(p3 * torch.ones(n_covariates, output_dim).to(device), is_binary=True)

            # for the approx posterior
            self.qlogits_z = torch.nn.Parameter(3.0 * torch.ones(1, output_dim).to(device))
            self.qlogits_c = torch.nn.Parameter(3.0 * torch.ones(n_covariates, output_dim).to(device))
            self.qlogits_cz = torch.nn.Parameter(2.0 * torch.ones(n_covariates, output_dim).to(device))


    def forward_z(self, z):
        value = self.mapping_z(z)
        if self.has_feature_level_sparsity:
            w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_z)
            return w * value
        else:
            return value

    def forward_c(self, c):
        if self.has_feature_level_sparsity:
            out = 0.0
            if self.training:
                w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_c)
            else:
                w = torch.sigmoid(self.qlogits_c)
            for j in range(self.n_covariates):
                out += w[j, :] * (self.mappings_c[j](c[:, j:(j + 1)]))
        else:
            out = sum(
                [(self.mappings_c[j](c[:, j:(j + 1)])) for j in range(self.n_covariates)]
            )

        return out

    def forward_cz(self, z, c):
        if self.has_feature_level_sparsity:
            out = 0.0
            if self.training:
                w = rsample_RelaxedBernoulli(self.temperature, self.qlogits_cz)
            else:
                w = torch.sigmoid(self.temperature, self.qlogits_cz)
            for j in range(self.n_covariates):
                input = torch.cat([z, c[:, j:(j + 1)]], dim=1)
                out += w[j, :] * (self.mappings_cz[j](input))
        else:
            out = sum(
                [(self.mappings_cz[j](torch.cat([z, c[:, j:(j + 1)]], dim=1))) for j in range(self.n_covariates)]
            )

        return out


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

        w_c = rsample_RelaxedBernoulli(self.temperature, self.qlogits_c)
        int_c = [
            # has shape [1, output_dim]
            w_c[j, :] * (self.mappings_c[j](self.grid_c[j]).mean(dim=0).reshape(1, self.output_dim)) for j in
            range(self.n_covariates)
        ]

        # create empty lists for int_cz1 and int_cz2
        int_cz_dc = [None] * self.n_covariates
        int_cz_dz = [None] * self.n_covariates

        w_cz = rsample_RelaxedBernoulli(self.temperature, self.qlogits_cz)
        for j in range(self.n_covariates):
            m1 = self.n_grid_z
            m2 = self.n_grid_c[j]

            out = w_cz[j, :] * self.mappings_cz[j](self.grid_cz[j])
            out = out.reshape(m1, m2, self.output_dim)

            # has shape [m1, output_dim]
            int_cz_dc[j] = out.mean(dim=1)
            # has shape [m2, output_dim]
            int_cz_dz[j] = out.mean(dim=0)

        return int_z, int_c, int_cz_dc, int_cz_dz

    def calculate_integrals_numpy(self):

        with torch.no_grad():
            w_z = rsample_RelaxedBernoulli(self.temperature, self.qlogits_z)

            int_z = (w_z * self.forward_z(self.grid_z).mean(dim=0).reshape(1, self.output_dim)).cpu().numpy()

            w_c = rsample_RelaxedBernoulli(self.temperature, self.qlogits_c)

            int_c = np.vstack([
                # has shape [1, output_dim]
                (w_c[j, :] * self.mappings_c[j](self.grid_c[j]).mean(dim=0).reshape(1, self.output_dim)).cpu().numpy() for j in
                range(self.n_covariates)
            ])

            # create empty lists for int_cz1 and int_cz2
            int_cz_dc = [None] * self.n_covariates
            int_cz_dz = [None] * self.n_covariates

            w_cz = rsample_RelaxedBernoulli(self.temperature, self.qlogits_cz)
            for j in range(self.n_covariates):
                m1 = self.n_grid_z
                m2 = self.n_grid_c[j]

                out = w_cz[j, :] * self.mappings_cz[j](self.grid_cz[j])
                out = out.reshape(m1, m2, self.output_dim)

                # has shape [m1, output_dim]
                int_cz_dc[j] = out.mean(dim=1).cpu().numpy()
                # has shape [m2, output_dim]
                int_cz_dz[j] = out.mean(dim=0).cpu().numpy()

            array_int_cz_1 = np.vstack(int_cz_dc)
            array_int_cz_2 = np.vstack(int_cz_dz)
            array_int_cz = np.vstack([array_int_cz_1, array_int_cz_2])

            return int_z, int_c, array_int_cz


    def calculate_penalty(self):
        int_z, int_c, int_cz_1, int_cz_2 = self.calculate_integrals()

        # penalty with fixed lambda0
        if self.penalty_type in ["fixed", "MDMM"]:
            penalty0 = self.lambda0 * (int_z.abs().mean())

            for j in range(self.n_covariates):
                penalty0 += self.lambda0 * (int_c[j].abs().mean() + int_cz_1[j].abs().mean() + int_cz_2[j].abs().mean())

        if self.penalty_type in ["BDMM", "MDMM"]:
            penalty_BDMM = (self.Lambda_z * int_z).mean()

            for j in range(self.n_covariates):
                penalty_BDMM += (self.Lambda_c[j] * int_c[j]).mean() + \
                               (self.Lambda_cz_1[j] * int_cz_1[j]).mean() + (self.Lambda_cz_2[j] * int_cz_2[j]).mean()


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

        neg_loglik = - self.loglik(y_pred, y_obs)

        if self.has_feature_level_sparsity:
            KL1 = approximate_KLqp(self.logits_z, self.qlogits_z)
            KL2 = approximate_KLqp(self.logits_c, self.qlogits_c)
            KL3 = approximate_KLqp(self.logits_cz, self.qlogits_cz)
            neg_loglik += 1.0 * (KL1 + KL2 + KL3)

        return neg_loglik, penalty, int_z, int_c, int_cz_dc, int_cz_dz

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
