import numpy as np
from typing import Optional


class PowerLawFourierSynthesisNeurons:
    def __init__(
        self,
        latent_dim: int,
        num_neurons: int,
        alpha: float,
        seed: Optional[int] = None,
        freq_max: int = 1000,
    ):
        self.d = latent_dim
        self.n = num_neurons
        self.alpha = alpha
        self.freq_max = freq_max

        self.feature_projection = None
        self.feature_frequency = None
        self.phase_shift = None
        self.proj_features_to_neurons = None
        self.init_embedding_functions(seed)

    def init_embedding_functions(self, seed: Optional[int] = 123456):
        rng = np.random.default_rng(seed)

        # Each 'feature' will be some sinusoidal function of a 1D projection of z and at a given
        # frequency. We'll use uniform sampling of frequencies and power-law decaying amplitudes.
        self.feature_projection = rng.standard_normal(size=(self.freq_max, self.d))
        self.feature_projection /= np.linalg.norm(self.feature_projection, axis=1, keepdims=True)
        self.feature_frequency = np.pi * rng.uniform(1, self.freq_max, size=self.freq_max)
        # Each neuron will read out a random phase. Represent the random phase using a unit complex
        # number per neuron per feature.
        self.phase_shift = rng.uniform(0, 2 * np.pi, size=(self.freq_max, self.n))
        self.phase_shift = np.exp(1j * self.phase_shift)
        # Decaying amplitude spectrum to get the desired power-law behavior of variances. Since
        # we want variance to go like freq**(-alpha), we need to scale the amplitude
        # by 1/freq**(-alpha/2)
        self.proj_features_to_neurons = rng.standard_normal(size=(self.freq_max, self.n))
        self.proj_features_to_neurons *= self.feature_frequency[:, None] ** (-self.alpha / 2)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Given (m, d) array of latent z, return (m, n) array of neural responses."""

        # TODO - consider some sources of noise like (1) additive noise on the final step, (2) noise
        #  in the 'readout' of z, etc

        # Compute the feature projections
        feature_projections = z @ self.feature_projection.T * self.feature_frequency

        # Calculate sinusoidal features using complex exponential so that we can phase shift per
        # neuron later without using more memory.
        features = np.exp(1j * feature_projections)

        # Project features to neurons and phase shift all at once
        neural_activities = np.einsum(
            "mf,fn,fn->mn", features, self.phase_shift, self.proj_features_to_neurons
        ).real

        return neural_activities


class RadialBasisNeurons:
    def __init__(
        self,
        latent_dim: int,
        num_neurons: int,
        bandwidth: float,
        seed: Optional[int] = None,
    ):
        self.d = latent_dim
        self.n = num_neurons
        self.bw = bandwidth

        self.centers = None
        self.init_embedding_functions(seed)

    def init_embedding_functions(self, seed: Optional[int] = 123456):
        rng = np.random.default_rng(seed)

        self.centers = rng.uniform(-1, 1, size=(self.n, self.d))

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Given (m, d) array of latent z, return (m, n) array of neural responses."""

        # TODO - consider some sources of noise like (1) additive noise on the final step, (2) noise
        #  in the 'readout' of z, etc

        dist_to_center = np.linalg.norm(z[:, None, :] - self.centers[None, :, :], axis=-1)
        activations = np.exp(-0.5 * (dist_to_center / self.bw) ** 2)

        return activations