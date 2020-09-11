import torch
from torch import nn
import torch.functional as F


class VAE(nn.Module):
    """ Basic Variational Autoencoder module. """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Params
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Functions
        self.leaky_relu = nn.LeakyReLU(0.1)

        # Net
        self.efc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.efc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mu = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.sigma = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.dfc1 = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.dfc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dfc3 = nn.Linear(hidden_dim, input_dim, bias=False)

    def encode(self, x):
        h1 = self.leaky_relu(self.efc1(x))
        h2 = self.leaky_relu(self.efc2(h1))
        return self.mu(h2), self.sigma(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.leaky_relu(self.dfc1(z))
        h2 = self.leaky_relu(self.dfc2(h1))
        return self.dfc3(h2)

    def forward(self, x):
        # Encode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Decode
        decoded = self.decode(z)

        return decoded, mu, logvar

