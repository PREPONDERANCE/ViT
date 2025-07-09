import mlx.core as mx

from mlx import nn


class MLPBlock(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int, act_layer: type[nn.Module]):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.Linear(hidden_dim, emb_dim),
            act_layer(),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers(x)


class DyT(nn.Module):
    def __init__(self, dim: int, channel_last: bool = False, alpha: float = 0.5):
        super().__init__()

        self.alpha = mx.array(mx.ones(1) * alpha)
        self.weight = mx.array(mx.ones(dim))
        self.bias = mx.array(mx.zeros(dim))

        self.channel_last = channel_last

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.tanh(self.alpha * x)
        if self.channel_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[None, None, :] + self.bias[None, None, :]
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = mx.array(mx.ones(dim))
        self.bias = mx.array(mx.zeros(dim))

    def __call__(self, x: mx.array) -> mx.array:
        u = x.mean(axis=-1, keepdims=True)
        a = mx.power((x - u), 2).mean(1, keepdims=True)
        x = (x - u) / mx.sqrt(a + self.eps)
        x = x * self.weight[None, None, :] + self.bias[None, None, :]
        return x
