import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, act_layer: type[nn.Module]):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Linear(hidden_dim, emb_dim),
            act_layer(),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DyT(nn.Module):
    def __init__(self, dim: int, channel_last: bool = False, alpha: float = 0.5):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1) * alpha)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

        self.channel_last = channel_last

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.alpha * x)
        if self.channel_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=1, keepdim=True)
        a = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / (a + self.eps).sqrt()

        return x * self.weight[:, None, None] + self.bias[:, None, None]
