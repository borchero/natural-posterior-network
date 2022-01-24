import math
from typing import cast, List, Tuple
import torch
from torch import nn
import natpn.distributions as D
from .model import NaturalPosteriorNetworkModel
from .output import Output


class NaturalPosteriorEnsembleModel(nn.Module):
    """
    Implementation of the NatPE model.
    """

    def __init__(self, networks: List[NaturalPosteriorNetworkModel]):
        """
        Args:
            networks: The NatPN networks whose outputs are combined by the ensemble. They or may
                not have equal configuration.
        """
        super().__init__()
        self.networks = nn.ModuleList(networks)

    @property
    def output(self) -> Output:
        """
        Returns the output module of the ensemble.
        """
        return cast(NaturalPosteriorNetworkModel, self.networks[0]).output

    def forward(self, x: torch.Tensor) -> Tuple[D.Posterior, torch.Tensor]:
        """
        Performs a Bayesian update over the target distribution for each input independently via
        Bayesian combination of the underlying networks' predictions. The returned posterior
        distribution carries all information about the prediction.

        Args:
            x: The inputs that are first passed to the encoder.

        Returns:
            The posterior distribution for every input.
        """
        outputs = [
            cast(NaturalPosteriorNetworkModel, network).posterior_update(x)
            for network in self.networks
        ]
        updates = [output[0] for output in outputs]
        log_probs = [output[1] for output in outputs]

        update = D.mixture_posterior_update(updates)
        posterior = cast(NaturalPosteriorNetworkModel, self.networks[0]).output.prior.update(
            update
        )
        log_prob = torch.stack(log_probs).logsumexp(0) - math.log(len(self.networks))
        return posterior, log_prob
