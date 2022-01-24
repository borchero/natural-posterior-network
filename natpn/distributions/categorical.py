from __future__ import annotations
import math
import torch
from ._base import ConjugatePrior, Likelihood, Posterior, PosteriorPredictive, PosteriorUpdate


class DirichletPrior(ConjugatePrior):
    """
    Dirichlet distribution as the conjugate prior of the Categorical likelihood.
    """

    def __init__(self, num_categories: int, evidence: float):
        """
        Args:
            num_categories: The number of categories for the Dirichlet distribution.
            evidence: The per-category evidence.
        """
        super().__init__(torch.ones(num_categories) / num_categories, torch.as_tensor(evidence))

    def update(self, update: PosteriorUpdate) -> Posterior:
        update_alpha = update.sufficient_statistics * update.log_evidence.exp().unsqueeze(-1)
        return Dirichlet(update_alpha + self.sufficient_statistics * self.evidence)


class Dirichlet(Posterior):
    """
    Dirichlet distribution as the posterior of the Categorical likelihood.
    """

    def __init__(self, alpha: torch.Tensor):
        self.alpha = alpha

    def expected_log_likelihood(self, data: torch.Tensor) -> torch.Tensor:
        a0 = self.alpha.sum(-1)
        a_true = self.alpha.gather(-1, data.unsqueeze(-1)).squeeze(-1)
        return a_true.digamma() - a0.digamma()

    def entropy(self) -> torch.Tensor:
        k = self.alpha.size(-1)
        a0 = self.alpha.sum(-1)

        # Approximate for large a0
        t1 = 0.5 * (k - 1) + 0.5 * (k - 1) * math.log(2 * math.pi)
        t2 = 0.5 * self.alpha.log().sum(-1)
        t3 = (k - 0.5) * a0.log()
        approx = t1 + t2 - t3

        # Calculate exactly for lower a0
        t1 = self.alpha.lgamma().sum(-1) - a0.lgamma() - (k - a0) * a0.digamma()
        t2 = ((self.alpha - 1) * self.alpha.digamma()).sum(-1)
        exact = t1 - t2

        return torch.where(a0 >= 10000, approx, exact)

    def maximum_a_posteriori(self) -> Likelihood:
        return self._map()

    def posterior_predictive(self) -> PosteriorPredictive:
        return self._map()

    def _map(self) -> Categorical:
        return Categorical(self.alpha.log() - self.alpha.sum(-1, keepdim=True).log())


class Categorical(Likelihood, PosteriorPredictive):
    """
    Categorical distribution for modeling discrete observations from a set of classes.
    """

    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def mean(self) -> torch.Tensor:
        return self.logits.argmax(-1)

    def uncertainty(self) -> torch.Tensor:
        return -(self.logits * self.logits.exp()).sum(-1)

    def expected_sufficient_statistics(self) -> torch.Tensor:
        return self.logits.exp()

    def symmetric_confidence_level(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
