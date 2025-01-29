import torch
from torch import nn
from typing import Optional
from abc import ABC, abstractmethod


class DataGenBase(nn.Module, ABC):
    def __init__(
        self,
        latent_dim: int,
        num_neurons: int,
        poisson_scale: Optional[float] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_neurons = num_neurons
        self.poisson_scale = poisson_scale
        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(seed)

    @abstractmethod
    def tuning(self, z: torch.Tensor) -> torch.Tensor:
        """Get (m, n) neural responses, normalized to [0,1] each, to (m, d) latent."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError("z must have shape (m, d)")
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z.shape[1] must match latent_dim ({self.latent_dim}) but is ({z.shape[1]})"
            )

        activations = self.tuning(z)

        if self.poisson_scale is None or self.poisson_scale < 1e-9:
            return activations
        else:
            return torch.poisson(self.poisson_scale * activations, generator=self.rng)


class PowerLawFourierSynthesisNeurons(DataGenBase):
    def __init__(
        self,
        alpha: float,
        latent_dim: int,
        num_neurons: int,
        freq_max: int = 1000,
        poisson_scale: Optional[float] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(latent_dim, num_neurons, poisson_scale, seed, device)
        self.alpha = alpha
        self.freq_max = freq_max

        self.feature_projection = None
        self.feature_frequency = None
        self.phase_shift = None
        self.proj_features_to_neurons = None
        self.init_embedding_functions()

    def init_embedding_functions(self):
        # Each 'feature' will be some sinusoidal function of a 1D projection of z and at a given
        # frequency. We'll use uniform sampling of frequencies and power-law decaying amplitudes.
        self.feature_projection = torch.randn(
            size=(self.freq_max, self.latent_dim), device=self.rng.device, generator=self.rng
        )
        self.feature_projection /= torch.linalg.norm(self.feature_projection, dim=1, keepdims=True)
        self.feature_frequency = torch.rand(
            size=(self.freq_max,), device=self.rng.device, generator=self.rng
        )
        self.feature_frequency = self.feature_frequency * torch.pi * (self.freq_max - 1) + 1
        # Each neuron will read out a random phase. Represent the random phase using a unit complex
        # number per neuron per feature.
        self.phase_shift = (
            torch.rand(
                size=(self.freq_max, self.num_neurons),
                device=self.rng.device,
                generator=self.rng,
            )
            * 2
            * torch.pi
        )
        self.phase_shift = torch.exp(1j * self.phase_shift)
        # Decaying amplitude spectrum to get the desired power-law behavior of variances. Since
        # we want variance to go like freq**(-alpha), we need to scale the amplitude
        # by 1/freq**(-alpha/2)
        self.proj_features_to_neurons = torch.randn(
            size=(self.freq_max, self.num_neurons),
            device=self.rng.device,
            generator=self.rng,
        )
        self.proj_features_to_neurons *= self.feature_frequency[:, None] ** (-self.alpha / 2)

        # Wrap all tuning parameters as nn.Parameter objects so that their device will be
        # automatically managed by torch.
        self.feature_projection = nn.Parameter(self.feature_projection)
        self.feature_frequency = nn.Parameter(self.feature_frequency)
        self.phase_shift = nn.Parameter(self.phase_shift)
        self.proj_features_to_neurons = nn.Parameter(self.proj_features_to_neurons)

    def tuning(self, z: torch.Tensor) -> torch.Tensor:
        # Compute the feature projections
        feature_projections = z @ self.feature_projection.T * self.feature_frequency

        # Calculate sinusoidal features using complex exponential so that we can phase shift per
        # neuron later without using more memory.
        features = torch.exp(1j * feature_projections)

        # Project features to neurons and phase shift all at once
        activations = torch.einsum(
            "mf,fn,fn->mn", features, self.phase_shift, self.proj_features_to_neurons + 0j
        ).real

        return torch.nn.functional.softplus(activations)


class RadialBasisNeurons(DataGenBase):
    def __init__(
        self,
        latent_dim: int,
        num_neurons: int,
        bandwidth: float,
        poisson_scale: Optional[float] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(latent_dim, num_neurons, poisson_scale, seed, device)
        self.bw = bandwidth

        self.centers = None
        self.init_embedding_functions()

    def init_embedding_functions(self):
        self.centers = torch.rand(
            size=(self.num_neurons, self.latent_dim),
            device=self.rng.device,
            generator=self.rng,
        )
        self.centers = self.centers * 2.0 - 1.0

        # Wrap all tuning parameters as nn.Parameter objects so that their device will be
        # automatically managed by torch.
        self.centers = nn.Parameter(self.centers)

    def tuning(self, z: torch.Tensor) -> torch.Tensor:
        dist_to_center = torch.linalg.norm(z[:, None, :] - self.centers[None, :, :], dim=-1)
        activations = torch.exp(-0.5 * (dist_to_center / self.bw) ** 2)

        return activations
