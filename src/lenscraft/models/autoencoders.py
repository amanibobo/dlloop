"""Anomaly detectors from Alexander et al. 2021, Appendix B, Tables IV-VI (DESIGN.md Section 7.2).

Ported layer for layer. The paper specifies a 150x150 grayscale input, for which the conv stack
produces 64 x 9 x 9 = 5184 features. The only generalisation here is that the flattened size and
the decoder's output paddings are computed from ``input_size`` instead of hard-coded, so the same
code also runs at other sizes where the arithmetic closes (e.g. 96, 150, 204). Sizes where it
does not (e.g. 64) raise at construction time; resize the images instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_sizes(input_size: int) -> tuple[int, int, int]:
    s1 = (input_size + 2 - 7) // 3 + 1  # conv 7, stride 3, pad 1
    s2 = (s1 + 2 - 7) // 3 + 1  # conv 7, stride 3, pad 1
    s3 = s2 - 6  # conv 7, stride 1, no pad
    return s1, s2, s3


def conv_feature_size(input_size: int) -> int:
    """Flattened feature size after the shared conv stack (5184 at the paper's 150 px)."""
    _, _, s3 = _conv_sizes(input_size)
    if s3 < 1:
        raise ValueError(f"input_size {input_size} is too small for the Table IV conv stack (paper uses 150)")
    return 64 * s3 * s3


def _decoder_paddings(input_size: int) -> tuple[int, int]:
    s1, s2, _ = _conv_sizes(input_size)
    op2 = s1 - (3 * s2 + 2)  # ConvTranspose2d(k=7, s=3, p=1): out = 3*(in-1) - 2 + 7 + op
    op3 = input_size - (3 * s1 + 1)  # ConvTranspose2d(k=6, s=3, p=1): out = 3*(in-1) - 2 + 6 + op
    if not (0 <= op2 <= 2 and 0 <= op3 <= 2):
        raise ValueError(
            f"input_size {input_size} cannot be reconstructed exactly by the Table IV decoder "
            f"(needs output_padding {op2}, {op3}); use 150 (paper) or another size such as 96 or 204"
        )
    return op2, op3


class _ConvStack(nn.Module):
    """Shared encoder conv layers (Tables IV-VI)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 7, stride=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 7, stride=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x).flatten(1)


class DCAEEncoder(nn.Module):
    """Table IV encoder: conv stack -> linear -> BatchNorm."""

    def __init__(self, latent_dim: int = 1000, input_size: int = 150) -> None:
        super().__init__()
        self.convs = _ConvStack()
        self.fc = nn.Linear(conv_feature_size(input_size), latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.fc(self.convs(x)))


class DCAEDecoder(nn.Module):
    """Table IV decoder (shared by DCAE, VAE and AAE)."""

    def __init__(self, latent_dim: int = 1000, input_size: int = 150) -> None:
        super().__init__()
        _, _, s3 = _conv_sizes(input_size)
        op2, op3 = _decoder_paddings(input_size)
        self.s3 = s3
        self.fc = nn.Linear(latent_dim, 64 * s3 * s3)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 7)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 7, stride=3, padding=1, output_padding=op2)
        self.deconv3 = nn.ConvTranspose2d(16, 1, 6, stride=3, padding=1, output_padding=op3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 64, self.s3, self.s3)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return torch.tanh(self.deconv3(x))


class DCAE(nn.Module):
    def __init__(self, latent_dim: int = 1000, input_size: int = 150) -> None:
        super().__init__()
        self.encoder = DCAEEncoder(latent_dim, input_size)
        self.decoder = DCAEDecoder(latent_dim, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class VAEEncoder(nn.Module):
    """Table V encoder: same conv stack, two linear heads (mu, logvar)."""

    def __init__(self, latent_dim: int = 1000, input_size: int = 150) -> None:
        super().__init__()
        self.convs = _ConvStack()
        n = conv_feature_size(input_size)
        self.fc_mu = nn.Linear(n, latent_dim)
        self.fc_logvar = nn.Linear(n, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.convs(x)
        return self.fc_mu(h), self.fc_logvar(h)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 1000, input_size: int = 150) -> None:
        super().__init__()
        self.encoder = VAEEncoder(latent_dim, input_size)
        self.decoder = DCAEDecoder(latent_dim, input_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction from the posterior mean (used for anomaly scoring)."""
        mu, _ = self.encoder(x)
        return self.decoder(mu)


class AAEEncoder(nn.Module):
    """Table VI encoder: same conv stack, linear, no BatchNorm."""

    def __init__(self, latent_dim: int = 1000, input_size: int = 150) -> None:
        super().__init__()
        self.convs = _ConvStack()
        self.fc = nn.Linear(conv_feature_size(input_size), latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.convs(x))


class AAEDiscriminator(nn.Module):
    """Table VI discriminator on the latent code."""

    def __init__(self, latent_dim: int = 1000) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AAE(nn.Module):
    def __init__(self, latent_dim: int = 1000, input_size: int = 150) -> None:
        super().__init__()
        self.encoder = AAEEncoder(latent_dim, input_size)
        self.decoder = DCAEDecoder(latent_dim, input_size)
        self.discriminator = AAEDiscriminator(latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_anomaly_model(arch: str, latent_dim: int = 1000, input_size: int = 150) -> nn.Module:
    if arch == "dcae":
        return DCAE(latent_dim, input_size)
    if arch == "vae":
        return VAE(latent_dim, input_size)
    if arch == "aae":
        return AAE(latent_dim, input_size)
    raise ValueError(f"unknown anomaly arch {arch!r}; expected 'dcae', 'vae' or 'aae'")
