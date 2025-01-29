import numpy as np
from typing import Optional
from abc import ABC, abstractmethod


class DataGenBase(ABC):
    def __init__(
        self,
        latent_dim: int,
        num_neurons: int,
        poisson_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.latent_dim = latent_dim
        self.num_neurons = num_neurons
        self.poisson_scale = poisson_scale
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def tuning(self, z: np.ndarray) -> np.ndarray:
        """Get (m, n) neural responses, normalized to [0,1] each, to (m, d) latent."""

    def __call__(self, z: np.ndarray) -> np.ndarray:
        if z.ndim != 2:
            raise ValueError("z must have shape (m, d)")
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z.shape[1] must match latent_dim ({self.latent_dim}) but is ({z.shape[1]})"
            )

        activations = self.tuning(z)

        if self.poisson_scale is None:
            return activations
        else:
            return self.rng.poisson(self.poisson_scale * activations)


class PowerLawFourierSynthesisNeurons(DataGenBase):
    def __init__(
        self,
        alpha: float,
        latent_dim: int,
        num_neurons: int,
        freq_max: int = 1000,
        poisson_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(latent_dim, num_neurons, poisson_scale, seed)
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
        self.feature_projection = self.rng.standard_normal(size=(self.freq_max, self.d))
        self.feature_projection /= np.linalg.norm(self.feature_projection, axis=1, keepdims=True)
        self.feature_frequency = np.pi * self.rng.uniform(1, self.freq_max, size=self.freq_max)
        # Each neuron will read out a random phase. Represent the random phase using a unit complex
        # number per neuron per feature.
        self.phase_shift = self.rng.uniform(0, 2 * np.pi, size=(self.freq_max, self.n))
        self.phase_shift = np.exp(1j * self.phase_shift)
        # Decaying amplitude spectrum to get the desired power-law behavior of variances. Since
        # we want variance to go like freq**(-alpha), we need to scale the amplitude
        # by 1/freq**(-alpha/2)
        self.proj_features_to_neurons = self.rng.standard_normal(size=(self.freq_max, self.n))
        self.proj_features_to_neurons *= self.feature_frequency[:, None] ** (-self.alpha / 2)

    def tuning(self, z: np.ndarray) -> np.ndarray:
        # Compute the feature projections
        feature_projections = z @ self.feature_projection.T * self.feature_frequency

        # Calculate sinusoidal features using complex exponential so that we can phase shift per
        # neuron later without using more memory.
        features = np.exp(1j * feature_projections)

        # Project features to neurons and phase shift all at once
        activations = np.einsum(
            "mf,fn,fn->mn", features, self.phase_shift, self.proj_features_to_neurons
        ).real

        return activations


class RadialBasisNeurons(DataGenBase):
    def __init__(
        self,
        latent_dim: int,
        num_neurons: int,
        bandwidth: float,
        poisson_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(latent_dim, num_neurons, poisson_scale, seed)
        self.bw = bandwidth

        self.centers = None
        self.init_embedding_functions()

    def init_embedding_functions(self):
        self.centers = self.rng.uniform(-1, 1, size=(self.num_neurons, self.latent_dim))

    def tuning(self, z: np.ndarray) -> np.ndarray:
        dist_to_center = np.linalg.norm(z[:, None, :] - self.centers[None, :, :], axis=-1)
        activations = np.exp(-0.5 * (dist_to_center / self.bw) ** 2)

        return activations
