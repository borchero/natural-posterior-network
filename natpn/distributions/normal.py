import math
import scipy.stats as st  # type: ignore
import torch
from natpn.utils import chunk_squeeze_last
from ._base import ConjugatePrior, Likelihood, Posterior, PosteriorPredictive, PosteriorUpdate


class StudentT(PosteriorPredictive):
    """
    Student's t-distribution as the posterior predictive of the Normal likelihood.
    """

    def __init__(self, df: torch.Tensor, loc: torch.Tensor, precision: torch.Tensor):
        self.df = df
        self.loc = loc
        self.precision = precision

    def mean(self) -> torch.Tensor:
        return self.loc

    def uncertainty(self) -> torch.Tensor:
        lbeta = torch.lgamma(0.5 * self.df) + math.lgamma(0.5) - torch.lgamma(0.5 * (self.df + 1))
        dfp1 = 0.5 * (self.df + 1)

        t1 = 0.5 * self.df.log() + lbeta - 0.5 * self.precision.log()
        t2 = dfp1 * (torch.digamma(dfp1) - torch.digamma(0.5 * self.df))
        return t1 + t2

    def symmetric_confidence_level(self, data: torch.Tensor) -> torch.Tensor:
        assert (
            not torch.is_grad_enabled()
        ), "The confidence level cannot currently track the gradient."

        probs_numpy = st.t.cdf(  # type: ignore
            data.cpu().numpy(),
            df=self.df.cpu().numpy(),
            loc=self.loc.cpu().numpy(),
            scale=self.precision.reciprocal().sqrt().cpu().numpy(),
        )
        probs = torch.from_numpy(probs_numpy).to(data.device)
        return 2 * (probs - 0.5).abs()


class NormalGammaPrior(ConjugatePrior):
    """
    Normal gamma distribution as the conjugate prior of the Normal likelihood.
    """

    def __init__(self, mean: float, scale: float, evidence: float):
        """
        Args:
            mean: The expected mean of the outputs.
            scale: The expected scale of the outputs.
            evidence: The certainty for the expectationon mean and scale.
        """
        super().__init__(
            torch.as_tensor([mean, mean ** 2 + scale ** 2]), torch.as_tensor(evidence)
        )

    def update(self, update: PosteriorUpdate) -> Posterior:
        prior_z1, prior_z2 = self.sufficient_statistics
        z1, z2 = chunk_squeeze_last(update.sufficient_statistics)

        prior_evidence = self.evidence
        evidence = update.log_evidence.exp()

        # Initialize Normal Gamma parameters
        posterior_evidence = evidence + prior_evidence
        posterior_z1 = (z1 * evidence + prior_z1 * prior_evidence) / posterior_evidence
        posterior_z2 = (z2 * evidence + prior_z2 * prior_evidence) / posterior_evidence

        mu = posterior_z1
        lambd = posterior_evidence
        alpha = 0.5 * posterior_evidence
        beta = 0.5 * posterior_evidence * (posterior_z2 - posterior_z1 ** 2)

        return NormalGamma(mu, lambd, alpha, beta)


class NormalGamma(Posterior):
    """
    Normal gamma distribution as the posterior of the Normal likelihood.
    """

    def __init__(
        self, mu: torch.Tensor, lambd: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
    ):
        self.mu = mu
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta

    def expected_log_likelihood(self, data: torch.Tensor) -> torch.Tensor:
        const = -math.log(2 * math.pi)
        scaled_mse = self.alpha / self.beta * (self.mu - data) ** 2
        diff = self.alpha.digamma() - self.beta.log() - self.lambd.reciprocal()
        return 0.5 * (const - scaled_mse + diff)

    def entropy(self) -> torch.Tensor:
        t1 = 0.5 + 0.5 * math.log(2 * math.pi)
        t2_exact = self.alpha.lgamma() + self.alpha - (self.alpha + 1.5) * self.alpha.digamma()
        t2_approx = t1 - 2 * self.alpha.log()
        t2 = torch.where(self.alpha >= 10000, t2_approx, t2_exact)
        t3 = 1.5 * self.beta.log() - 0.5 * self.lambd.log()
        return t1 + t2 + t3

    def maximum_a_posteriori(self) -> Likelihood:
        return Normal(self.mu, self.alpha / self.beta)

    def posterior_predictive(self) -> PosteriorPredictive:
        # This is empirically wrong but seems to be the correct theoretical derivation
        precision = (self.alpha * self.lambd) / (self.beta * (self.lambd + 1))
        # precision = (self.alpha * self.lambd) / self.beta
        return StudentT(2 * self.alpha, self.mu, precision)


class Normal(Likelihood):
    """
    Normal distribution for modeling continuous data.
    """

    def __init__(self, loc: torch.Tensor, precision: torch.Tensor):
        self.loc = loc
        self.precision = precision

    def mean(self) -> torch.Tensor:
        return self.loc

    def uncertainty(self) -> torch.Tensor:
        return 0.5 - 0.5 * math.log(2 * math.pi) * self.precision.log()

    def expected_sufficient_statistics(self) -> torch.Tensor:
        return torch.stack([self.loc, self.loc ** 2 + self.precision.reciprocal()], dim=-1)
