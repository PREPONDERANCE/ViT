# import mlx.core as mx


# def random_mask(x: mx.array, mask_ratio: float) -> tuple[mx.array, mx.array, mx.array]:
#     B, H, W, C = x.shape
#     L = H * W

#     x = x.reshape(B, -1, C)

#     len_keep = int(L * (1 - mask_ratio))

#     noise = mx.random.normal((B, L))

#     idx_shuffle = mx.argsort(noise, axis=-1)
#     idx_restore = mx.argsort(idx_shuffle, axis=-1)

#     idx_keep = idx_shuffle[:, :len_keep]
#     idx_keep = mx.expand_dims(idx_keep, -1)
#     idx_keep = mx.repeat(idx_keep, C, axis=-1)

#     x_masked = mx.take_along_axis(x, idx_keep, axis=1)
#     Hr = Wr = int(len_keep**0.5)
#     x_masked = x_masked.reshape(B, Hr, Wr, C)

#     mask = mx.array(mx.ones((B, L)))
#     mask[:, :len_keep] = 0
#     mask = mx.take_along_axis(mask, idx_restore, axis=1)

#     return x_masked, mask, idx_restore


# a, b, c = random_mask(mx.random.normal((32, 64, 64, 256)), 0.75)

# print(a.shape)
# print(b.shape)
# print(c.shape)


# print(b)


import torch
from vit_torch.common import DyT


d = DyT(256)

out = d(torch.randn(1, 256))

print(out.shape)
