import torch
import torch.nn as nn
import tqdm

from distillation.algorithms.base import SamplingDirection, SamplingMethod, euler_sampling, flow_matching_loss, heun_sampling


class DDIB:
    def __init__(self, noisy_model: nn.Module, clean_model: nn.Module, lr: float, seed: int):
        self.noisy_model = noisy_model
        self.clean_model = clean_model
        self.lr = lr
        self.seed = seed

        self.noisy_optim = None
        self.clean_optim = None
        self.rng = None

    def init_training(self, device):
        self.noisy_optim = torch.optim.Adam(
            self.noisy_model.parameters(), lr=self.lr
        )
        self.clean_optim = torch.optim.Adam(
            self.clean_model.parameters(), lr=self.lr
        )
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(self.seed)

    def reset_training(self):
        self.noisy_model.reset_weights() # type: ignore
        self.clean_model.reset_weights() # type: ignore
        self.noisy_optim = None
        self.clean_optim = None
        self.rng = None

    def train_single_model(self, dataloader, n_steps: int, device: torch.device | str, model: nn.Module, optim: torch.optim.Optimizer, model_name: str):
        model = model.to(device)

        for _ in tqdm.tqdm(range(n_steps), desc=f"training {model_name}", dynamic_ncols=True, leave=False):
            model.train()
            data = next(dataloader)
            data = data.to(device)
            noise = torch.empty_like(data, device=device).normal_(generator=self.rng)
            loss = flow_matching_loss(
                model, noise_samples=noise, data_samples=data, generator=self.rng
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
        optim.zero_grad()

    def train_noisy(self, dataloader, n_steps: int, device: torch.device | str):
        self.noisy_model = self.noisy_model.to(device)
        if self.noisy_optim is None:
            self.init_training(device=device)
        assert self.noisy_model is not None and self.noisy_optim is not None
        self.train_single_model(dataloader, n_steps, device, self.noisy_model, self.noisy_optim, "noisy model")

    def train_clean(self, dataloader, n_steps: int, device: torch.device | str):
        self.clean_model = self.clean_model.to(device)
        if self.clean_optim is None:
            self.init_training(device=device)
        assert self.clean_model is not None and self.clean_optim is not None
        self.train_single_model(dataloader, n_steps, device, self.clean_model, self.clean_optim, "clean model")

    def restore(
        self,
        noisy_samples: torch.Tensor,
        n_steps: int,
        device: torch.device | str,
        sampling_method: SamplingMethod = SamplingMethod.EULER
    ):
        noisy_samples = noisy_samples.to(device)
        self.noisy_model.eval()
        self.clean_model.eval()
        if sampling_method == SamplingMethod.EULER:
            sampling_fn = euler_sampling
        elif sampling_method == SamplingMethod.HEUN:
            sampling_fn = heun_sampling
        else:
            raise ValueError(sampling_method)
        latents = sampling_fn(
            self.noisy_model,
            n_steps=n_steps,
            x0=noisy_samples,
            direction=SamplingDirection.DATA_TO_NOISE
        )
        cleans = sampling_fn(
            self.clean_model,
            n_steps=n_steps,
            x0=latents[-1],
            direction=SamplingDirection.NOISE_TO_DATA
        )
        return latents + cleans

    def sample(
        self,
        n_steps: int,
        sample_size: tuple[int, ...],
        device: torch.device | str,
        rng: torch.Generator,
        model: torch.nn.Module,
        sampling_method: SamplingMethod,
    ):
        model = model.to(device)
        model.eval()
        x0 = torch.randn(sample_size, generator=rng, device=device)
        if sampling_method == SamplingMethod.EULER:
            samples = euler_sampling(model, n_steps, x0=x0)
        elif sampling_method == SamplingMethod.HEUN:
            samples = heun_sampling(model, n_steps, x0=x0)
        else:
            raise ValueError(sampling_method)
        return samples

    def sample_clean(
        self,
        n_steps: int,
        sample_size: tuple[int, ...],
        device: torch.device | str,
        rng: torch.Generator,
        sampling_method: SamplingMethod = SamplingMethod.EULER,
    ):
        return self.sample(n_steps, sample_size, device, rng, self.clean_model, sampling_method)

    def sample_noisy(
        self,
        n_steps: int,
        sample_size: tuple[int, ...],
        device: torch.device | str,
        rng: torch.Generator,
        sampling_method: SamplingMethod = SamplingMethod.EULER,
    ):
        return self.sample(n_steps, sample_size, device, rng, self.noisy_model, sampling_method)
