import torch
from torch.distributions import MixtureSameFamily, MultivariateNormal, Categorical


class MixGauss:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        means = torch.tensor([[4.0, 0.0], [5.0, 1.0], [4.5, 2.0]])
        covs = torch.stack([torch.eye(2), torch.eye(2), torch.eye(2)], dim=0) / 50
        # means = torch.tensor([[0.0, 2.0]])
        # covs = torch.stack([torch.eye(2)], dim=0) / 10
        self.distribution = MixtureSameFamily(
            mixture_distribution=Categorical(probs=torch.tensor([0.333, 0.333, 0.333])),
            component_distribution=MultivariateNormal(means, covs)
        )

    def sample(self):
        return self.distribution.sample(sample_shape=(self.batch_size, ))

    def get_dataloader(self):
        while True:
            yield self.sample()