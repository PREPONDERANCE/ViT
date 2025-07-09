# import mlx.core as mx

# from vit_mlx import MaskedAutoEncoderViT


# mae = MaskedAutoEncoderViT(1024, 16)
# img = mx.random.normal((4, 1024, 1024, 3))

# out = mae(img)
# print(out)

import torch
from vit_torch import MaskedAutoEncoderViT

mae = MaskedAutoEncoderViT(1024)
img = torch.randn(2, 3, 1024, 1024)

out = mae(img)
print(out)
