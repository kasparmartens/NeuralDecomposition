import torch
import torch.nn as nn
import torch
import numpy as np

from math import ceil

from .helpers import KL_standard_normal

class CVAE(nn.Module):
    """
    CVAE with Neural Decomposition as part of the decoder
    """

    def __init__(self, encoder, decoder, lr, device="cpu"):
        super().__init__()

        self.encoder = encoder

        self.decoder = decoder

        self.output_dim = self.decoder.output_dim

        # optimizer for NN pars and likelihood noise
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.device = device

        self.to(device)


    def forward(self, data_subset, beta=1.0, device="cpu"):
        # we assume data_subset containts two elements
        Y, c = data_subset
        Y, c = Y.to(device), c.to(device)

        # encode
        mu_z, sigma_z = self.encoder(Y, c)
        eps = torch.randn_like(mu_z)
        z = mu_z + sigma_z * eps

        # decode
        y_pred = self.decoder.forward(z, c)
        decoder_loss, penalty, int1, int2, int3, int4 = self.decoder.loss(y_pred, Y)

        # loss function
        VAE_KL_loss = KL_standard_normal(mu_z, sigma_z)

        # Note that when this loss (neg ELBO) is calculated on a subset (minibatch),
        # we should scale it by data_size/minibatch_size, but it would apply to all terms
        total_loss = decoder_loss + beta * VAE_KL_loss

        return total_loss, int1, int2, int3, int4

    def calculate_test_loglik(self, Y, c):
        """
        maps (Y, x) to z and calculates p(y* | x, z_mu)
        :param Y:
        :param c:
        :return:
        """
        mu_z, sigma_z = self.encoder(Y, c)

        Y_pred = self.decoder.forward(mu_z, c)

        return self.decoder.loglik(Y_pred, Y)


    def optimize(self, data_loader, augmented_lagrangian_lr, n_iter=50000, logging_freq=20, logging_freq_int=100, temperature_start=4.0, temperature_end=0.2, lambda_start=None, lambda_end=None, verbose=True):

        # sample size
        N = len(data_loader.dataset)

        # number of iterations = (numer of epochs) * (number of iters per epoch)
        n_epochs = ceil(n_iter / len(data_loader))
        if verbose:
            print(f"Fitting Neural Decomposition.\n\tData set size {N}. # iterations = {n_iter} (i.e. # epochs <= {n_epochs})\n")

        loss_values = np.zeros(ceil(n_iter // logging_freq))

        if self.decoder.has_feature_level_sparsity:
            temperature_grid = torch.linspace(temperature_start, temperature_end, steps=n_iter // 10, device=self.device)

        if lambda_start is None:
            lambda_start = self.decoder.lambda0
            lambda_end = self.decoder.lambda0
        lambda_grid = torch.linspace(lambda_start, lambda_end, steps=n_iter // 10, device=self.device)

        # get shapes for integrals
        _int_z, _int_c, _int_cz = self.decoder.calculate_integrals_numpy()
        # log the integral values
        n_logging_steps = ceil(n_iter // logging_freq_int)
        int_z_values = np.zeros([n_logging_steps, _int_z.shape[0], self.output_dim])
        int_c_values = np.zeros([n_logging_steps, _int_c.shape[0], self.output_dim])
        int_cz_values = np.zeros([n_logging_steps, _int_cz.shape[0], self.output_dim])

        iteration = 0
        for epoch in range(n_epochs):

            for batch_idx, data_subset in enumerate(data_loader):

                if iteration >= n_iter:
                    break

                loss, int_z, int_c, int_cz_dc, int_cz_dz = self.forward(data_subset, beta=1.0, device=self.device)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.decoder.has_feature_level_sparsity:
                    self.decoder.set_temperature(temperature_grid[iteration // 10])
                self.decoder.lambda0 = lambda_grid[iteration // 10]

                # update for BDMM
                with torch.no_grad():
                    self.decoder.Lambda_z += augmented_lagrangian_lr * int_z
                    self.decoder.Lambda_c += augmented_lagrangian_lr * int_c
                    self.decoder.Lambda_cz_1 += augmented_lagrangian_lr * int_cz_dc
                    self.decoder.Lambda_cz_2 += augmented_lagrangian_lr * int_cz_dz

                # logging for the loss function
                if iteration % logging_freq == 0:
                    index = iteration // logging_freq
                    loss_values[index] = loss.item()

                # logging for integral constraints
                if iteration % logging_freq_int == 0:
                    int_z, int_c, int_cz = self.decoder.calculate_integrals_numpy()

                    index = iteration // logging_freq_int
                    int_z_values[index, :] = int_z
                    int_c_values[index, :] = int_c
                    int_cz_values[index, :] = int_cz

                if verbose and iteration % 500 == 0:
                    print(f"\tIter {iteration:5}.\tTotal loss {loss.item():.3f}")

                iteration += 1

        # collect all integral values into one array
        integrals = np.hstack([int_z_values, int_c_values, int_cz_values]).reshape(n_iter // logging_freq_int, -1).T

        return loss_values, integrals

class CVAE_with_fixed_z(CVAE):
    """
    Same as the above CVAE class, but assuming a fixed latent variable z, thus effectively only training the decoder.
    We assume z is given by the data_loader, i.e. we assume it returns tuples (Y, c, z)
    """

    def __init__(self, decoder, lr):
        super().__init__(encoder=None, decoder=decoder, lr=lr)

    def forward(self, data_subset, beta=1.0):
        # we assume data_subset containts three elements
        Y, c, z = data_subset

        # decoding step
        y_pred = self.decoder.forward(z, c)
        decoder_loss, penalty, int1, int2, int3, int4 = self.decoder.loss(y_pred, Y)

        # no KL(q(z) | p(z)) term because z fixed
        total_loss = decoder_loss

        return total_loss, int1, int2, int3, int4
