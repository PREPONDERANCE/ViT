import time
import mlx.core as mx

from vit_mlx import VisionTransformer, DyT

s = time.perf_counter()

PATCH = 16
B, H, W, C = 32, 1024, 1024, 3
vit = VisionTransformer(
    H, PATCH, depth=6, norm_layer=DyT, global_attn_indexes=(3, 6, 9)
)

img = mx.random.normal([B, H, W, C])

print(vit(img).shape)

e = time.perf_counter()

print(f"total time: {e - s}")
