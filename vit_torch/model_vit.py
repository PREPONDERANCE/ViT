import torch
import torch.nn.functional as F


from torch import nn

from .common import MLPBlock, LayerNorm2d


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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        emb_dim: int = 768,
        num_heads: int = 12,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: tuple[int, int] | None = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = emb_dim // num_heads
        self.scale = head_dim ** (-0.5)

        self.to_qkv = nn.Linear(emb_dim, emb_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(emb_dim, emb_dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            H, W = input_size
            self.rel_h = nn.Parameter(torch.ones(2 * H - 1, head_dim))
            self.rel_w = nn.Parameter(torch.ones(2 * W - 1, head_dim))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def _get_rel_pos(
        self,
        q_size: int,
        k_size: int,
        rel_pos: torch.Tensor,
    ) -> torch.Tensor:
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)

        rel_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
        return rel_pos[rel_coords.long()]

    def _add_rel_pose(
        self,
        q: torch.Tensor,
        attn: torch.Tensor,
        q_shape: tuple[int, int],
        k_shape: tuple[int, int],
    ):
        q_h, q_w = q_shape
        k_h, k_w = k_shape

        rel_h = self._get_rel_pos(q_h, k_h, self.rel_h)
        rel_w = self._get_rel_pos(q_w, k_w, self.rel_w)

        B, _, C = q.shape
        q = q.reshape(B, q_h, q_w, C)
        rel_h = torch.einsum("bhwc, hkc -> bhwk", q, rel_h)
        rel_w = torch.einsum("bhwc, hkc -> bhwk", q, rel_w)

        attn = (
            attn.view(B, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, None]
            + rel_w[:, :, :, None, :]
        ).view(B, q_h * q_w, k_h * k_w)

        return attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        qkv: torch.Tensor = self.to_qkv(x)
        qkv = (
            qkv.reshape(B, H * W, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
            .reshape(3, B * self.num_heads, H * W, -1)
        )
        q, k, v = qkv.chunk(3, dim=0)

        attn: torch.Tensor = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = self._add_rel_pose(q, attn, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )

        return self.proj(x)


class Block(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
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
            emb_dim=emb_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(emb_dim)
        self.mlp = MLPBlock(emb_dim, int(emb_dim * mlp_ratio), act_layer)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def _window_partition(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        ws = self.window_size
        B, H, W, C = x.shape

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = x.shape[1], x.shape[2]

        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        window = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)

        return window, (Hp, Wp)

    def _window_undo_partition(
        self, x: torch.Tensor, pad: tuple[int, int], hw: tuple[int, int]
    ) -> torch.Tensor:
        ws = self.window_size
        H, W = hw
        Hp, Wp = pad

        B = x.shape[0] // (Hp * Wp // ws // ws)
        x = x.view(B, Hp // ws, Wp // ws, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, Hp, Wp, -1)

        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm: torch.Tensor = self.norm1(x)

        if self.window_size:
            _, H, W, _ = x.shape
            x_norm, pad = self._window_partition(x_norm)

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
        patch_size: int = 16,
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

        self.patch_emb = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=(0, 0),
            in_chan=in_chan,
            emb_dim=emb_dim,
        )

        self.pos_emb: torch.Tensor = None
        if use_abs_pos:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, emb_dim)
            )

        self.blocks = [
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
            for i in range(depth)
        ]

        self.neck = nn.Sequential(
            nn.Conv2d(emb_dim, out_chan, kernel_size=1, stride=1),
            LayerNorm2d(out_chan),
            nn.Conv2d(out_chan, out_chan, kernel_size=1, stride=1),
            LayerNorm2d(out_chan),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_emb(x)
        if self.pos_emb is not None:
            x = x + self.pos_emb

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x)
        x = x.permute(0, 3, 1, 2)

        return x
