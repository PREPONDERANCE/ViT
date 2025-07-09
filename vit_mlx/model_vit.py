import mlx.core as mx

from mlx import nn

from .common import MLPBlock, DyT


class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chan: int = 3,
        emb_dim: int = 768,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_chan,
            emb_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Attention(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: tuple[int, int] | None = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = emb_dim // num_heads
        self.scale = head_dim ** (-0.5)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            H, W = input_size
            self.rel_h = mx.array(mx.zeros(2 * H - 1, head_dim))
            self.rel_w = mx.array(mx.zeros(2 * W - 1, head_dim))

        self.to_qkv = nn.Linear(emb_dim, 3 * emb_dim, qkv_bias)
        self.proj = nn.Linear(emb_dim, emb_dim)

    def _get_rel_pos(self, q_size: int, k_size: int, rel_pos: mx.array) -> mx.array:
        q_range = mx.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_range = mx.arange(k_size)[None, :] * max(q_size / k_size, 1.0)

        coords = (q_range - k_range) + (k_size - 1) * max(q_size / k_size, 1.0)
        return rel_pos[coords.astype(mx.int64)]

    def _add_rel_pos(
        self,
        q: mx.array,
        attn: mx.array,
        q_shape: tuple[int, int],
        k_shape: tuple[int, int],
    ) -> mx.array:
        q_h, q_w = q_shape
        k_h, k_w = k_shape

        rel_h = self._get_rel_pos(q_h, k_h, self.rel_h)
        rel_w = self._get_rel_pos(q_w, k_w, self.rel_w)

        b = q.shape[0]
        q = q.reshape(b, q_h, q_w, -1)
        rel_h = mx.einsum("bhwc, hkc -> bhwk", q, rel_h)
        rel_w = mx.einsum("bhwc, hkc -> bhwk", q, rel_w)

        attn = (
            attn.reshape(b, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, None]
            + rel_w[:, :, :, None, :]
        ).view(b, q_h * q_w, k_h * k_w)

        return attn

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, _ = x.shape
        qkv = (
            self.to_qkv(x)
            .reshape(B, H * W, 3, self.num_heads, -1)
            .transpose(2, 0, 3, 1, 4)
        )
        # (3, B, heads, H*W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).split(3, axis=0)
        # (B*heads, H*W, C)
        attn = mx.matmul(q * self.scale, k.transpose(0, 1, 3, 2))

        if self.use_rel_pos:
            attn = self._add_rel_pos(q, attn, (H, W), (H, W))

        attn = mx.softmax(attn, axis=-1)
        # (B*heads, H*W, C)
        x = (
            mx.matmul(attn, v)
            .reshape(B, self.num_heads, H, W, -1)
            .transpose(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )

        return self.proj(x)


class Block(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
    ):
        super().__init__()

        self.window_size = window_size

        self.norm1 = norm_layer(emb_dim)
        self.attn = Attention(
            emb_dim,
            num_heads,
            qkv_bias,
            use_rel_pos,
            input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(emb_dim)
        self.mlp = MLPBlock(emb_dim, int(emb_dim * mlp_ratio), act_layer)

    def _window_partition(
        self, x: mx.array, window_size: int
    ) -> tuple[mx.array, tuple[int, int]]:
        B, H, W, C = x.shape

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        if pad_h > 0 or pad_w > 0:
            x = mx.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))

        Hp, Wp = pad_h + H, pad_w + W

        x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        x = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)

        return x, (Hp, Wp)

    def _window_undo_partition(
        self, x: mx.array, pad: tuple[int, int], hw: tuple[int, int]
    ) -> mx.array:
        H, W = hw
        Hp, Wp = pad
        _, window_size, _, C = x.shape

        x = x.reshape(
            -1, Hp // window_size, Wp // window_size, window_size, window_size, C
        )
        x = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, Hp, Wp, C)

        if Hp > H or Wp > W:
            x = x[:, :H, :W, :]

        return x

    def __call__(self, x: mx.array) -> mx.array:
        x_norm = self.norm1(x)

        if self.window_size:
            _, H, W, _ = x.shape
            x_norm, pad = self._window_partition(x_norm, self.window_size)

        x = self.attn(x_norm) + x

        if self.window_size:
            x = self._window_undo_partition(x, pad, (H, W))

        x_norm = self.norm2(x)
        x = self.mlp(x_norm) + x

        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chan: int = 3,
        emb_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chan: int = 256,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chan=in_chan,
            emb_dim=emb_dim,
        )

        self.pos_emb: mx.array = None
        if use_abs_pos:
            self.pos_emb = mx.array(
                mx.zeros(
                    shape=(1, img_size // patch_size, img_size // patch_size, emb_dim)
                )
            )

        self.blocks: list[Block] = []
        for i in range(depth):
            self.blocks.append(
                Block(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_rel_pos=use_rel_pos,
                    window_size=window_size if i in global_attn_indexes else 0,
                    input_size=(img_size // patch_size, img_size // patch_size),
                )
            )

        self.neck = nn.Sequential(
            nn.Conv2d(emb_dim, out_chan, kernel_size=1, bias=False),
            DyT(out_chan, channel_last=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False),
            DyT(out_chan, channel_last=True),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.patch_embed(x)
        if self.pos_emb is not None:
            x = x + self.pos_emb

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x)
        return x
