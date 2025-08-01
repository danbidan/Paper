import torch
import torch.nn as nn


class EEGEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)  # log σ²
        return mu, logvar


class EEGDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        out = self.fc2(h)
        return out


def reparameterize(mu, logvar, n_samples=100):
    """Reparameterization trick for sampling from latent space"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(n_samples, *mu.shape)
    return mu.unsqueeze(0) + std.unsqueeze(0) * eps 