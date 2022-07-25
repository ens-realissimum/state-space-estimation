from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from typing import NoReturn, Tuple, Union

import numpy as np
from scipy.stats import gamma, multivariate_normal

import bayesian_framework.core.covariance_utils as cov_utils
import utils.matrix_utils as matrix_utils
from .covariance_type import CovarianceType
from .noise_type import NoiseType


class GeneralStochasticModel(ABC):
    """
    The generated noise data structure (model), Model has the following required fields.
    Depending on the noise source type, the data structure may also have other type dependent fields.
        .update (function handle) <<optional>> function to update the internal structure of noise source;
        .covarianceType (string) .
    """

    @property
    @abstractmethod
    def noise_type(self) -> NoiseType:
        """
        Noise source type (NoiseType enum).
        :return: name of noise type.
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Noise source dimension (scalar number).
        :return: noise dimension.
        """
        pass

    @property
    @abstractmethod
    def covariance_type(self) -> CovarianceType:
        """
        Type of covariance (CovarianceType enum).
        :return: Type of covariance
        """
        pass

    @property
    @abstractmethod
    def covariance(self) -> np.ndarray:
        """
        Covariance matrix. None if covariance is not applicable for particular problem
        :return: Covariance matrix (numpy.matrix).
        """
        pass

    @property
    @abstractmethod
    def mean(self) -> np.ndarray:
        """
        Expectation, mathematical expectation, EV, average, mean value, mean, or first moment.
        :return: 'Expected value' (first moment) np.array or scalar number.
        """
        pass

    @abstractmethod
    def sample(self, size: int) -> np.ndarray:
        """
        Method to generate N noise source samples.
        :type size: int
        :return: nd-array of noise source samples with dimension equal to "dimension" x N.
        """
        pass

    @abstractmethod
    def likelihood(self, samples: np.ndarray) -> np.ndarray:
        """
        Method to evaluate the likelihood of a given noise source sample.
        :type samples: np.array
        :return: likelihood of a given noise source sample (scalar number).
        """
        pass

    def update(self, **kwargs) -> NoReturn:
        pass

    @property
    def n_components(self) -> int:
        return 0

    @property
    def weights(self) -> Union[np.ndarray, None]:
        return None


class GammaStochasticModel(GeneralStochasticModel):
    def __init__(self, *, shape, scale):
        is_shape_list = isinstance(shape, collections.Sequence)
        is_scale_list = isinstance(scale, collections.Sequence)

        if is_shape_list and is_scale_list:
            if len(shape) != len(scale):
                raise Exception("dimensions mismatch of shape and scale")

        if is_shape_list and not is_scale_list or not is_shape_list and is_scale_list:
            raise Exception("shape and scale both should be sequence or scalar at the same time")

        self._dimension = len(shape) if is_shape_list else 1
        self._shape = shape
        self._scale = scale

    def __str__(self) -> str:
        return f"Gamma process. Shape: {self._shape}; Scale: {self._scale}"

    def likelihood(self, samples: np.ndarray) -> np.ndarray:
        if samples is None:
            raise Exception("<samples> should be n-dim numpy array")

        return np.atleast_1d(
            gamma.pdf(samples.T, self._shape, scale=self._scale)
        )

    def sample(self, size: int) -> np.matrix:
        if size is None:
            raise Exception("<size> is mandatory but was not specified")

        if size <= 0:
            raise Exception("<size> is negative or equal to zero. Must be integer greater than zero")

        return np.matrix(
            gamma.rvs(self._shape, scale=self._scale, size=size).T
        )

    @property
    def mean(self) -> np.ndarray:
        return np.atleast_1d(
            gamma.mean(self._shape, scale=self._scale)
        )

    @property
    def covariance(self) -> np.matrix:
        return np.matrix(
            gamma.var(self._shape, scale=self._scale)
        )

    @property
    def covariance_type(self) -> CovarianceType:
        return CovarianceType.full

    @property
    def dim(self) -> int:
        return self._dimension

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.gamma


class GaussianStochasticModel(GeneralStochasticModel):
    def __init__(self, *, mean, covariance, covariance_type):
        if isinstance(mean, collections.Sequence):
            if not matrix_utils.is_square_2d_array(covariance):
                raise Exception("Covariance matrix must be square matrix")

            if len(mean) != matrix_utils.shape_square_2d_array(covariance):
                raise Exception("Dimensions mismatch of mean and covariance matrix")

            self._dimension = len(mean)
            self._mean = np.asarray(mean)
        else:
            self._dimension = 1
            self._mean = np.atleast_1d(mean)

        is_diag = covariance_type in (CovarianceType.diag, CovarianceType.sqrt_diag)
        self._covariance_type = covariance_type
        self._covariance = np.matrix(
            matrix_utils.ensure_diagonal_matrix(covariance) if is_diag else covariance
        )
        self._covariance_sqrt = np.matrix(
            cov_utils.to_sqrt_covariance(covariance, covariance_type)
        )
        self._covariance_full = np.matrix(
            cov_utils.to_full_covariance(covariance, covariance_type)
        )

    def __str__(self) -> str:
        return f"Gaussian noise. Mean: {self.mean}; Covariance: {self._covariance_full}"

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.gaussian

    @property
    def dim(self) -> int:
        return self._dimension

    @property
    def covariance_type(self) -> CovarianceType:
        return self._covariance_type

    @property
    def covariance(self) -> np.matrix:
        return self._covariance

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    def sample(self, size: int) -> np.matrix:
        return np.matrix(
            multivariate_normal.rvs(mean=self._mean, cov=self._covariance_full, size=size).T
        )

    def likelihood(self, samples: np.ndarray) -> np.matrix:
        if samples is None:
            raise Exception(f"<samples> should be {self.dim}-dim numpy array")

        return np.matrix(
            multivariate_normal.pdf(samples.T, mean=self._mean, cov=self._covariance_full)
        )


class ComboGaussianStochasticModel(GeneralStochasticModel):
    def __init__(self, *, dimension, sources):
        if not isinstance(sources, collections.Sequence):
            raise Exception("sources must be sequence")

        cov_type = sources[0].covariance_type
        if not all(source.covariance_type == cov_type for source in sources):
            raise Exception("all source items must have same covariance_type")

        cov = matrix_utils.put_matrices_into_zero_matrix_one_by_one(
            dimension,
            list(map(lambda x: x.covariance, sources))
        )
        self._dimension = dimension
        self._covariance_type = sources[0].covariance_type
        self._mean = np.atleast_1d([sub_mean for source in sources for sub_mean in source.mean])
        self._covariance = np.atleast_2d(cov)
        self._covariance_full = np.atleast_2d(
            cov_utils.to_full_covariance(cov, cov_type)
        )
        self._sources = sources

        if not np.size(self._mean) == dimension:
            raise Exception("length of mean vector must be equal to dimension")

    def __str__(self) -> str:
        return f"Combo Gaussian.\nMean:\n {self.mean};\nCovariance:\n {self._covariance_full}\n\n"

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.combo_gaussian

    @property
    def dim(self) -> int:
        return self._dimension

    @property
    def covariance_type(self) -> CovarianceType:
        return self._covariance_type

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    def sample(self, size: int) -> np.ndarray:
        return np.atleast_2d(
            multivariate_normal.rvs(loc=self._mean, scale=self._covariance_full, size=size).T
        )

    def likelihood(self, samples: np.ndarray) -> np.ndarray:
        return np.atleast_1d(
            multivariate_normal.pdf(samples.T, loc=self._mean, scale=self._covariance_full)
        )

    def update(self, **kwargs) -> NoReturn:
        shift = 0

        for source in self._sources:
            source.mean = self._mean[shift: shift + source.dim]
            source.covariance = self._covariance[shift: shift + source.dim, shift: shift + source.dim]

            shift += source.dim


class ComboStochasticModel(ComboGaussianStochasticModel):
    def __str__(self) -> str:
        return f"Combo noise.\nMean:\n {self.mean};\nCovariance:\n {self.covariance}\n\n"

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.combo

    def sample(self, size: int) -> np.ndarray:
        result = np.zeros((size, self.dim))
        shift = 0

        for source in self._sources:
            result[shift: shift + source.dim, :] = source.sample(size)
            shift += source.dim

        return np.atleast_2d(result)

    def likelihood(self, samples: np.ndarray) -> np.ndarray:
        _, size = np.shape(samples)
        llh = np.zeros(size)
        shift = 0

        for source in self._sources:
            llh = llh * source.likelihood(samples[shift: shift + source.dim, :])
            shift += source.dim

        return np.atleast_1d(llh)


class GaussianMixtureStochasticModel(GeneralStochasticModel):
    def __init__(self, *, mixture_size, mean, covariance, covariance_type, weights=None):
        self._n_components = mixture_size
        self._mean = np.atleast_2d(mean)
        self._covariance_type = covariance_type
        self._covariance = np.atleast_3d(covariance)
        cov_full = map(lambda x: cov_utils.to_full_covariance(x, covariance_type), covariance)
        self._covariance_full = np.atleast_3d(list(cov_full))
        self._dimension, _ = np.shape(mean)
        self._weights = np.ones(mixture_size) / mixture_size if weights is None else np.asarray(weights / sum(weights))

    def __str__(self) -> str:
        return f"Gaussian Mixture.\nWeights:\n {self.weights};\nMean:\n {self._mean};\nCovariance:\n {self._covariance_full}\n\n"

    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.gaussian_mixture

    @property
    def dim(self) -> int:
        return self._dimension

    @property
    def covariance_type(self) -> CovarianceType:
        return self._covariance_type

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def sample(self, size) -> np.ndarray:
        samples_component = np.random.multinomial(size, self.weights)
        return np.atleast_2d(
            np.hstack(
                [
                    np.transpose(np.random.multivariate_normal(mean, covariance, int(sample)))
                    for (mean, covariance, sample) in zip(self.mean, self.covariance, samples_component)
                ]
            )
        )

    def likelihood(self, samples: np.ndarray) -> np.ndarray:
        # evidence is used as likelihood, because
        # i-th entry of evidence is the total data probability for a given data vector X[i],
        # i.e. P(X[i]) = sum_over_all_j { P(X[i] | C[j]) }
        # i.e. effectively it's a likelihood
        _, _, llh, _ = self.probability(samples)
        return np.atleast_1d(llh)

    def probability(
        self,
        samples: np.ndarray,
        evidence_weights: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates any of the related (through Bayes rule) probabilities of a Gaussian Mixture Model and
        a given data set (samples). Probabilities are:
                   P(X|C) * P(C)                   likelihood * prior
        P(C|X) = -----------------   posterior =  --------------------
                       P(X)                            evidence

        where C is the component classes (Gaussian) of the GMM and X is the data.

        :param samples:
        :param evidence_weights:
        :return: Tuple of: Prior probability, Likelihood, Evidence, Posterior probability
            Prior       the prior (without seeing data) probability of a component density generating any given
                        data vector, i.e. P(C[i]). This is simply the same as the prior mixing weights
            Likelihood  matrix where the j, i-th entry is the likelihood of input column vector i (of samples)
                        conditioned on component density j, i.e. P(X[i] | C[j]);
            Evidence    matrix where the i-th entry is the total data probability for a given data vector X[i], i.e.
                        P(X[i]) = sum_over_all_j { P(X[i] | C[j]) };
            Posterior   matrix where the j, i-th entry is the posterior probability (after seeing the data) that
                        a component density j has generated a specific data vector X[i], i.e.
                        P(C[j] | X[i]) (class posterior probabilities).
        """
        prior = np.copy(self.weights)

        llh = np.column_stack(
            [
                multivariate_normal.pdf(samples.T, mean=mean, cov=cov)
                for (mean, cov) in zip(self.mean, self._covariance_full)
            ]
        )

        evidence_w = np.ones(np.shape(llh)) if evidence_weights is None else evidence_weights
        evidence = (llh / evidence_w) @ prior

        posterior = llh / (np.reciprocal(prior)[:, np.newaxis] @ evidence[np.newaxis, :]).T
        posterior = posterior / np.sum(posterior)

        return prior, llh, evidence, posterior

    @staticmethod
    def from_gaussian(gauss_process: GaussianStochasticModel) -> GaussianMixtureStochasticModel:
        return GaussianMixtureStochasticModel(
            mixture_size=1,
            mean=[gauss_process.mean],
            covariance=[gauss_process.covariance],
            covariance_type=gauss_process.covariance_type
        )


def build_stochastic_process(noise_type: NoiseType, **kwargs) -> GeneralStochasticModel:
    if noise_type == NoiseType.gamma:
        return GammaStochasticModel(**kwargs)
    elif noise_type == NoiseType.gaussian:
        return GaussianStochasticModel(**kwargs)
    elif noise_type == NoiseType.combo_gaussian:
        return ComboGaussianStochasticModel(**kwargs)
    elif noise_type == NoiseType.combo:
        return ComboStochasticModel(**kwargs)
    elif noise_type == NoiseType.gaussian_mixture:
        return GaussianMixtureStochasticModel(**kwargs)
    else:
        raise Exception(f"Not supported noise_type: {noise_type.name}")
