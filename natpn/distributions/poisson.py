from __future__ import annotations
import math
import scipy.stats as st  # type: ignore
import torch
from ._base import ConjugatePrior, Likelihood, Posterior, PosteriorPredictive, PosteriorUpdate


class NegativeBinomial(PosteriorPredictive):
    """
    Binomial distribution as the posterior predictive of the Poisson likelihood.
    """

    def __init__(self, num_failures: torch.Tensor, success_probability: torch.Tensor):
        self.num_failures = num_failures
        self.success_probability = success_probability

    def mean(self) -> torch.Tensor:
        return self.num_failures * self.success_probability / (1 - self.success_probability)

    def uncertainty(self) -> torch.Tensor:
        return self.num_failures * self.success_probability / (1 - self.success_probability) ** 2

    def symmetric_confidence_level(self, data: torch.Tensor) -> torch.Tensor:
        assert (
            not torch.is_grad_enabled()
        ), "The confidence level cannot currently track the gradient."

        # We need float64 due to values very close to 1 for the failure probability
        probs_numpy = st.nbinom.cdf(  # type: ignore
            data.cpu().double().numpy(),
            self.num_failures.cpu().double().numpy(),
            1 - self.success_probability.cpu().double().numpy(),
        )
        probs = torch.from_numpy(probs_numpy).to(data.device)
        return 2 * (probs - 0.5).abs()


class GammaPrior(ConjugatePrior):
    """
    Gamma distribution as the conjugate prior of the Poisson likelihood.
    """

    sufficient_statistics: torch.Tensor
    evidence: torch.Tensor

    def __init__(self, rate: float, evidence: float):
        super().__init__(torch.as_tensor(rate), torch.as_tensor(evidence))

    def update(self, update: PosteriorUpdate) -> Posterior:
        sufficient_statistics = update.sufficient_statistics
        evidence = update.log_evidence.exp()

        alpha = sufficient_statistics * evidence + self.sufficient_statistics * self.evidence
        beta = evidence + self.evidence
        return Gamma(alpha, beta)


class Gamma(Posterior):
    """
    Gamma distribution as the posterior of the Poisson likelihood.
    """

    def __init__(self, shape: torch.Tensor, rate: torch.Tensor):
        super().__init__()
        self.shape = shape
        self.rate = rate

    def expected_log_likelihood(self, data: torch.Tensor) -> torch.Tensor:
        t1 = data * (self.shape.digamma() - self.rate.log())
        t2 = self.shape / self.rate
        return t1 - t2

    def entropy(self) -> torch.Tensor:
        # Approximate for large alpha
        t1 = 0.5 + 0.5 * math.log(2 * math.pi)
        t2 = 0.5 * self.shape.log() - self.rate.log()
        approx = t1 + t2

        # Calculate exactly for smaller alpha
        t1 = self.shape - self.rate.log() + self.shape.lgamma()
        t2 = (1 - self.shape) * self.shape.digamma()
        exact = t1 + t2

        return torch.where(self.shape >= 10000, approx, exact)

    def maximum_a_posteriori(self) -> Likelihood:
        return Poisson(self.shape / self.rate)

    def posterior_predictive(self) -> PosteriorPredictive:
        return NegativeBinomial(self.shape, (1 + self.rate).reciprocal())


class Poisson(Likelihood):
    """
    Poisson distribution for modeling count data.
    """

    def __init__(self, rate: torch.Tensor):
        self.rate = rate

    def mean(self) -> torch.Tensor:
        return self.rate

    def uncertainty(self) -> torch.Tensor:
        return self.rate

    def expected_sufficient_statistics(self) -> torch.Tensor:
        return self.rate
