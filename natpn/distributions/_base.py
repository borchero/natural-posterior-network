from __future__ import annotations
from abc import ABC, abstractmethod
from typing import NamedTuple
import torch
from torch import nn


class PosteriorUpdate(NamedTuple):
    """
    The information for a posterior update.
    """

    #: The sufficient statistics of the likelihood, of shape ``[batch_shape, statistics_shape]``.
    sufficient_statistics: torch.Tensor
    #: The log-evidence of the likelihood's sufficient statistics, of shape ``[batch_shape]``.
    log_evidence: torch.Tensor


class _Distribution(ABC):
    @abstractmethod
    def mean(self) -> torch.Tensor:
        """
        Computes the mean of the posterior predictive.

        Returns:
            A tensor of shape ``[batch_shape]``.
        """

    @abstractmethod
    def uncertainty(self) -> torch.Tensor:
        """
        Computes the uncertainty of the distribution. If possible, this computes the entropy,
        otherwise it computes the variance.

        Returns:
            A tensor of shape ``[batch_shape]``.
        """


class Likelihood(_Distribution, ABC):
    """
    Base class for all distributions that my describe the likelihood of some data. Every likelihood
    distribution is required to have a conjugate prior.
    """

    @abstractmethod
    def expected_sufficient_statistics(self) -> torch.Tensor:
        """
        Computes the expected value of the sufficient statistics of the distribution.

        Returns:
            A tensor of shape ``[batch_shape, statistics_shape]`` with the expected values.
        """


class ConjugatePrior(nn.Module, ABC):
    """
    Base class for conjugate priors of likelihood distributions. The prior is meant to be included
    in modules. A Bayesian update can be performed to obtain a posterior. The conjugate prior is
    typically initialized from a prior guess on the sufficient statistic and a "certainty" value.
    """

    sufficient_statistics: torch.Tensor
    evidence: torch.Tensor

    def __init__(self, sufficient_statistics: torch.Tensor, evidence: torch.Tensor):
        super().__init__()
        self.register_buffer("sufficient_statistics", sufficient_statistics)
        self.register_buffer("evidence", evidence)

    @abstractmethod
    def update(self, update: PosteriorUpdate) -> Posterior:
        """
        Applies a Bayesian update using the provided update.

        Args:
            update: The update to apply, providing the sufficient statistics and the log-evidence.

        Returns:
            The posterior distribution.
        """


class Posterior(ABC):
    """
    Base class for posteriors of likelihood distributions.
    """

    @abstractmethod
    def expected_log_likelihood(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the expected log-likelihood of observing the provided data. The expectation is
        computed with respect to the distribution over the parameters of the likelihood function
        that is linked to this conjugate prior.

        Args:
            data: The data for which to compute the log-likelihood. The tensor must have shape
                ``[batch_shape, event_shape]``.

        Returns:
            A tensor provided the expected log-likelihood for all items in ``data``. The tensor has
            shape ``[batch_shape]``.
        """

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of this distribution.

        Returns:
            A tensor with shape ``[batch_shape]``.
        """

    @abstractmethod
    def maximum_a_posteriori(self) -> Likelihood:
        """
        Returns the a posteriori estimate with the most likely parameters of the likelihood
        distribution.

        Returns:
            The likelihood distribution with the same batch shape as this distribution.
        """

    @abstractmethod
    def posterior_predictive(self) -> PosteriorPredictive:
        """
        Returns the posterior predictive distribution obtained from the conjugate prior's
        parameters.

        Returns:
            The posterior predictive with the same batch shape as this distribution.
        """


class PosteriorPredictive(_Distribution, ABC):
    """
    Base class for posterior predictive distributions.
    """

    @abstractmethod
    def symmetric_confidence_level(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the symmetric confidence level for observing each of the provided data samples.
        The confidence level is the smallest level such that the confidence interval under the
        predictive posterior contains the data point. A lower confidence level, thus, indicates a
        more accurate prediction.

        Args:
            data: The data for which to obtain the confidence levels. The tensor must have shape
                ``[batch_shape, event_shape]``.

        Returns:
            A tensor provided the confidence levels for all items in ``data``. The tensor has
            shape ``[batch_shape]``.
        """
