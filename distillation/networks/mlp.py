import torch
import torch.nn as nn

__all__ = (
    "MLPUNet",
    "MLPUNetTimeCond",
    "MLPUNetTimesCond",
)


class MLPBase(nn.Module):
    def __init__(
        self,
        data_dim: int,
        n_layers: int,
        n_channels: int,
        time_dim: int,
        act: nn.Module = nn.SiLU(),
        norm: type[nn.Module] | None = nn.LayerNorm
    ):
        super().__init__()
        self.norm_layer = norm
        layers = [
            nn.Linear(data_dim + time_dim, n_channels, bias=True),
            act,
            self.get_normalization(n_channels)
        ]
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(n_channels, n_channels, bias=False),
                act,
                self.get_normalization(n_channels)
            ])
        layers.extend([
            nn.Linear(n_channels, data_dim, bias=True)
        ])
        self.mlp = nn.Sequential(*layers)
        self.reset_weights()

    def get_normalization(self, n_channels) -> nn.Module:
        if self.norm_layer is None:
            return nn.Identity()
        elif self.norm_layer is type[nn.LayerNorm]:
            return nn.LayerNorm(n_channels)
        else:
            raise NotImplementedError(f"Unsupported normalization {self.norm_layer}")

    def reset_weights(self, rng: torch.Generator | None = None):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu", generator=rng)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

class MLPUNet(MLPBase):
    def __init__(
        self,
        data_dim: int,
        n_layers: int,
        n_channels: int,
        act: nn.Module = nn.SiLU(),
        norm: type[nn.Module] | None = nn.LayerNorm
    ):
        super().__init__(
            data_dim=data_dim, n_layers=n_layers, n_channels=n_channels, time_dim=1, act=act, norm=norm)

    def forward(self, x):
        # x: B, C
        data = torch.cat([x, torch.ones_like(x)[:, :1]], dim=1)
        out = self.mlp(data)
        return out


class MLPUNetTimeCond(MLPBase):
    def __init__(
        self,
        data_dim: int,
        n_layers: int,
        n_channels: int,
        act: nn.Module = nn.SiLU(),
        norm: type[nn.Module] | None = nn.LayerNorm
    ):
        super().__init__(
            data_dim=data_dim, n_layers=n_layers, n_channels=n_channels, time_dim=1, act=act, norm=norm)

    def forward(self, x, t):
        # x: B, C
        # t: B, 1
        data = torch.cat([x, t.view(-1, 1)], dim=1)
        out = self.mlp(data)
        return out


class MLPUNetTimesCond(MLPBase):
    def __init__(
        self,
        data_dim: int,
        n_layers: int,
        n_channels: int,
        act: nn.Module = nn.SiLU(),
        norm: type[nn.Module] | None = nn.LayerNorm
    ):
        super().__init__(
            data_dim=data_dim, n_layers=n_layers, n_channels=n_channels, time_dim=2, act=act, norm=norm)

    def forward(self, x, t, s):
        # x: B, C
        # t: B, 1
        # s: B, 1
        data = torch.cat([
            x, t.view(-1, 1), s.view(-1, 1)
        ], dim=1)
        out = self.mlp(data)
        return out
