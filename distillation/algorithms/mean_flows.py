

import torch
import tqdm
from distillation.algorithms.base import Distiller

from ema_pytorch import EMA


class MeanFlow(Distiller):
    def __init__(
            self,
            model: torch.nn.Module,
            lr: float = 1e-4,
            optim_betas: tuple[float, float] = (0.9, 0.95),
            ema_beta: float = 0.9999,
            ema_first_upd: int = 100,
            flow_ratio: float = 0.75,
            norm_p: float = 1.0,
            norm_eps: float = 0.01,
            seed: int = 42,
        ):
        self.model = model
        # optimization
        if ema_beta > 0:
            self.ema_model = EMA(
                self.model,
                beta=ema_beta,
                update_after_step=ema_first_upd,
                update_every=10,
            )
        else:
            self.ema_model = None
        self.optim_betas = optim_betas
        self.lr_ = lr
        self.seed = seed
        self.optim = None
        self.rng = None
        # loss
        self.norm_eps = norm_eps
        self.norm_p = norm_p
        self.flow_ratio = flow_ratio

    @property
    def lr(self):
        return self.lr_

    @lr.setter
    def lr(self, new_lr):
        self.lr_ = new_lr
        if self.optim is not None:
            for group in self.optim.param_groups:
                group['lr'] = new_lr

    def init_training(self, device):
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=self.optim_betas
        )
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(self.seed)

    def pretrain(self, dataloader, n_steps: int, device):
        self.model = self.model.to(device)
        if self.ema_model is not None:
            self.ema_model = self.ema_model.to(device)
        if self.optim is None:
            self.init_training(device)
        assert self.optim is not None and self.rng is not None

        for _ in tqdm.tqdm(range(n_steps), desc="Pretraining MeanFlow", dynamic_ncols=True, leave=False):
            self.model.train()
            data = next(dataloader)
            data = data.to(device)
            loss = self.loss(data, device=device, rng=self.rng)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            if self.ema_model is not None:
                self.ema_model.update()

    def sample(
        self,
        n_steps: int,
        sample_size: tuple[int, ...],
        device: torch.device | str,
        rng: torch.Generator,
        model: torch.nn.Module | None = None,
    ):
        if model is None:
            model = self.ema_model if self.ema_model is not None else self.model
        t_vals = torch.linspace(0.0, 1.0, n_steps + 1).tolist()
        time_shape = (sample_size[0], *([1] * (len(sample_size) - 1)))
        xt = torch.randn(sample_size, device=device, generator=rng)
        traj = [xt.detach().cpu()]
        for i in range(n_steps):
            r = torch.full(time_shape, t_vals[i], device=device)
            t = torch.full(time_shape, t_vals[i + 1], device=device)
            v = model(xt, t, r)
            xt = xt + (t - r) * v
            traj.append(xt.detach().cpu())
        return traj

    def sample_t_r(self, batch_size: int, device: torch.device | str, rng: torch.Generator):
        t = torch.rand(batch_size, generator=rng, device=device)
        r = torch.rand(batch_size, generator=rng, device=device)
        t = torch.maximum(t, r)
        r = torch.minimum(t, r)

        # A proportion of time-steps used as in standard flow matching (r = t)
        num_selected = int(self.flow_ratio * batch_size)
        ind_selected = torch.randperm(batch_size, device=device, generator=rng)[:num_selected]
        r[ind_selected] = t[ind_selected]

        return t, r

    def loss(self, samples: torch.Tensor, device: torch.device | str, rng: torch.Generator):
        # samples: B, SSIZE
        batch_size = samples.shape[0]
        data_dims = samples.dim() - 1
        t, r = self.sample_t_r(batch_size, device, rng)
        t = t.view(-1, *([1] * data_dims))
        r = r.view(-1, *([1] * data_dims))

        x0 = torch.empty_like(samples).normal_(generator=rng)
        x1 = samples
        xr = (1 - r) * x0 + r * x1
        v = x1 - x0

        u, dudt = torch.func.jvp( # type: ignore
            self.model,
            primals=(xr, t, r),
            tangents=(v, torch.zeros_like(t), torch.ones_like(r))
        )
        u_tgt = v + torch.clamp(t - r, min=0.0, max=1.0) * dudt
        u_tgt = u_tgt.detach()

        # Square loss with adaptive weighting
        loss = torch.sum(
            (u - u_tgt) ** 2,
            dim=tuple(range(1, data_dims + 1))  # sum over pixels
        )
        adp_wt = (loss + self.norm_eps) ** self.norm_p
        loss = loss / adp_wt.detach()
        loss = loss.mean()  # average over batch
        return loss