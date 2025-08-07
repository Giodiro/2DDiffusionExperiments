import torch
import tqdm
from distillation.algorithms.base import PretrainedFlowDistiller, flow_matching_loss, sample_time


class DiffInstruct(PretrainedFlowDistiller):
    def __init__(
        self,
        pretr_model: torch.nn.Module,
        aux_model: torch.nn.Module,
        dist_model: torch.nn.Module,
        pretraining_lr: float,
        auxiliary_lr: float,
        distillation_lr: float,
        seed: int = 42,
    ):
        super().__init__(pretr_model, pretraining_lr, seed)
        self.aux_model = aux_model
        self.dist_model = dist_model
        self.auxiliary_lr = auxiliary_lr
        self.distillation_lr = distillation_lr
        self.aux_optim, self.dist_optim = None, None

    def init_distillation(self, device):
        self.aux_model.load_state_dict(self.pretr_model.state_dict())
        self.aux_optim = torch.optim.Adam(
            self.aux_model.parameters(), lr=self.auxiliary_lr
        )
        self.dist_optim = torch.optim.Adam(
            self.dist_model.parameters(), lr=self.distillation_lr
        )
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(self.seed)

    def reset_distillation(self):
        self.aux_model.reset_weights() # type: ignore
        self.dist_model.reset_weights() # type: ignore
        self.aux_optim = None
        self.dist_optim = None
        self.rng = None

    def distill(self, dataloader, n_steps: int, device: torch.device | str):
        self.pretr_model = self.pretr_model.to(device)
        self.aux_model = self.aux_model.to(device)
        self.dist_model = self.dist_model.to(device)
        if self.aux_optim is None or self.dist_optim is None:
            self.init_distillation(device)
        assert self.aux_optim is not None and self.dist_optim is not None
        self.pretr_model.eval()
        data = next(dataloader)  # only to get the correct sample shape
        for _ in tqdm.tqdm(range(n_steps), desc="Distilling DiffInstruct", dynamic_ncols=True, leave=False):
            # 1. Inner step (flow-matching)
            self.dist_model.eval()
            self.aux_model.train()
            self.aux_optim.zero_grad()
            noise_samples = torch.empty_like(data, device=device).normal_(generator=self.rng)
            generated = self.dist_model(noise_samples).detach()
            flow_loss = flow_matching_loss(
                self.aux_model, noise_samples=noise_samples, data_samples=generated, generator=self.rng
            )
            flow_loss.backward()
            self.aux_optim.step()
            self.aux_optim.zero_grad()
            # 2. Outer step (distillation)
            self.dist_optim.zero_grad()
            self.dist_model.train()
            self.aux_model.eval()
            noise_samples = torch.empty_like(data, device=device).normal_(generator=self.rng)
            sid_loss = self.loss(noise_samples)
            sid_loss.backward()
            self.dist_optim.step()
            self.dist_optim.zero_grad()

    def sample_distilled(self, sample_size: tuple[int, ...], device: torch.device | str, rng: torch.Generator):
        with torch.no_grad():
            model = self.dist_model.to(device)
            model.eval()
            x0 = torch.randn(sample_size, generator=rng, device=device)
            samples = model(x0)
            return [x0.cpu(), samples.cpu()]

    def loss(
        self,
        noise_samples: torch.Tensor,
    ):
        data_dims = tuple(range(1, noise_samples.dim()))
        gen_samples = self.dist_model(noise_samples)
        t = sample_time(gen_samples.shape, gen_samples.device, generator=self.rng, eps=1e-3)
        xt_gen = (1 - t) * noise_samples + t * gen_samples

        # TODO: Fix random state for v_teacher and v_auxiliary
        with torch.no_grad():
            v_teacher = self.pretr_model(xt_gen, t)
            v_auxiliary = self.aux_model(xt_gen, t)
            v_diff = v_auxiliary - v_teacher

        loss_batch = (v_diff * gen_samples).sum(dim=data_dims)
        loss_weight = ((1 - t) / t).squeeze() * torch.abs((gen_samples - noise_samples) - v_teacher).sum(dim=data_dims).clip(min=1e-5).detach()
        loss = torch.mean(loss_batch / loss_weight)
        return loss
