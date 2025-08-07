import abc
from enum import Enum
from typing import Literal

from torch import Tensor
import torch
import tqdm


class Models(Enum):
    FLOW_MATCHING = 1
    EDM = 2


class Distiller(abc.ABC):
    def __init__(self):
        pass

    def pretrain(self, dataloader, n_steps: int, device: torch.device | str):
        pass

    def distill(self):
        pass


class PretrainedFlowDistiller(Distiller, abc.ABC):
    def __init__(
        self,
        pretr_model: torch.nn.Module,
        pretraining_lr: float,
        seed: int = 42,
        model: Models = Models.FLOW_MATCHING,
    ):
        self.pretr_model = pretr_model
        self.pretraining_lr = pretraining_lr
        self.seed = seed
        self.pretr_optim = None
        self.rng = None

    def init_pretraining(self, device):
        self.pretr_optim = torch.optim.Adam(
            self.pretr_model.parameters(), lr=self.pretraining_lr
        )
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(self.seed)

    def reset_pretraining(self):
        self.pretr_model.reset_weights() # type: ignore
        self.pretr_optim = None
        self.rng = None

    def pretrain(self, dataloader, n_steps: int, device: torch.device | str):
        self.pretr_model = self.pretr_model.to(device)
        if self.pretr_optim is None:
            self.init_pretraining(device=device)
        assert self.pretr_optim is not None and self.rng is not None

        for _ in tqdm.tqdm(range(n_steps), desc="Pretraining Score Identity Distillation", dynamic_ncols=True, leave=False):
            self.pretr_model.train()
            data = next(dataloader)
            data = data.to(device)
            noise = torch.empty_like(data, device=device).normal_(generator=self.rng)
            loss = flow_matching_loss(
                self.pretr_model, noise_samples=noise, data_samples=data, generator=self.rng
            )
            self.pretr_optim.zero_grad()
            loss.backward()
            self.pretr_optim.step()
        self.pretr_optim.zero_grad()

    def sample_pretrained(
        self,
        n_steps: int,
        sample_size: tuple[int, ...],
        device: torch.device | str,
        rng: torch.Generator,
        model: torch.nn.Module | None = None,
    ):
        if model is None:
            model = self.pretr_model
        model = model.to(device)
        model.eval()
        x0 = torch.randn(sample_size, generator=rng, device=device)
        samples = euler_sampling(model, n_steps, x0=x0)
        return samples


def sample_time(
    sample_size: tuple[int, ...],
    device: torch.device | str,
    eps: float = 1e-3,
    distribution: Literal["uniform"] = "uniform",
    generator: torch.Generator | None = None,
) -> Tensor:
    batch_size = sample_size[0]
    extra_dims = len(sample_size) - 1
    if distribution == "uniform":
        return torch.rand(
            batch_size, *([1] * extra_dims), generator=generator, device=device
        ) * (1.0 - eps) + eps
    else:
        raise ValueError(distribution)


def edm_loss(
    model: torch.nn.Module,
    data_samples: Tensor,
    P_std,
    P_mean,
    sigma_data,
    generator: torch.Generator | None = None
):
    batch_size = data_samples.shape[0]
    extra_dims = data_samples.dim() - 1
    time_rnd = torch.randn(
        batch_size, *([1] * extra_dims),
        device=data_samples.device,
        generator=generator
    )
    sigma = torch.exp(time_rnd * P_std + P_mean)
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

    noise = torch.empty_like(data_samples).normal_(generator=generator) * sigma
    denoised = model(data_samples + noise, sigma)
    loss = weight * ((denoised - data_samples) ** 2)
    return loss


def flow_matching_loss(
    model: torch.nn.Module,
    noise_samples: Tensor,
    data_samples: Tensor,
    time_distribution: Literal["uniform"] = "uniform",
    generator: torch.Generator | None = None
):
    ode_time = sample_time(
        noise_samples.shape, noise_samples.device, distribution=time_distribution, generator=generator
    )

    xt = (1 - ode_time) * noise_samples + ode_time * data_samples
    true_velocity = data_samples - noise_samples
    pred_velocity = model(xt, ode_time)

    return torch.nn.functional.mse_loss(true_velocity, pred_velocity)


def time_to_tensor(t: float, sample: Tensor) -> Tensor:
    batch_size = sample.shape[0]
    sample_dims = sample.dim() - 1
    return torch.full(
        (batch_size, *([1] * sample_dims)),
        fill_value=t,
        dtype=sample.dtype,
        device=sample.device
    )


@torch.no_grad()
def euler_sampling(
    model: torch.nn.Module,
    n_steps: int,
    x0: Tensor,
) -> list[Tensor]:
    eps = 1e-3
    traj = [x0.cpu()]
    xt = x0
    dt = 1 / n_steps
    for i in range(n_steps):
        t = i / n_steps * (1.0 - eps) + eps
        velocity = model(xt, time_to_tensor(t, xt))
        xt = xt + velocity * dt
        traj.append(xt.cpu())
    return traj


@torch.no_grad()
def heun_sampling(
    model: torch.nn.Module,
    n_steps: int,
    x0: Tensor
):
    t_steps = torch.linspace(0, 1, steps=n_steps + 1).tolist()
    traj = [x0]
    x_next = x0
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        d_cur = model(x_cur, time_to_tensor(t_cur, x_cur))
        x_next = x_cur + (t_next - t_cur) * d_cur
        # 2nd order correction
        if i < n_steps - 1:
            d_prime = model(x_next, time_to_tensor(t_next, x_next))
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        traj.append(x_next)
    return traj