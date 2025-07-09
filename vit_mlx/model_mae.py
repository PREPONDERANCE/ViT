import mlx.core as mx

from mlx import nn
from .model_vit import VisionTransformer, Block


class MaskedAutoEncoderViT(VisionTransformer):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chan: int = 3,
        emb_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chan: int = 512,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        global_attn_indexes: tuple[int, ...] = (),
        decoder_num_heads: int = 8,
        decoder_depth: int = 8,
        mask_ratio: float = 0.75,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chan,
            emb_dim,
            depth,
            num_heads,
            mlp_ratio,
            out_chan,
            qkv_bias,
            norm_layer,
            act_layer,
            use_abs_pos,
            use_rel_pos,
            window_size,
            global_attn_indexes,
        )

        self.patch_size = patch_size

        self.blocks = self._construct_blocks(
            emb_dim=emb_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
            input_size=img_size // (patch_size / (1 - mask_ratio)),
            depth=depth,
        )

        self.decoder_pos_emb: mx.array = None
        if use_abs_pos:
            self.decoder_pos_emb = mx.array(
                mx.zeros(shape=(1, img_size // patch_size, img_size // patch_size, out_chan))
            )

        self.decoder = self._construct_blocks(
            emb_dim=out_chan,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
            input_size=img_size // patch_size,
            depth=decoder_depth,
        )

        self.decoder_neck = nn.Linear(out_chan, patch_size**2 * 3)

        self.mask_ratio = mask_ratio
        self.mask_token = mx.array(mx.zeros((1, 1, out_chan)))

    def _construct_blocks(
        self,
        emb_dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        norm_layer: type[nn.Module],
        act_layer: type[nn.Module],
        window_size: int,
        global_attn_indexes: tuple[int, ...],
        input_size: int,
        depth: int,
    ) -> list[nn.Module]:
        return [
            Block(
                emb_dim=emb_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=False,
                window_size=window_size if i in global_attn_indexes else 0,
                input_size=(input_size, input_size),
            )
            for i in range(depth)
        ]

    def random_mask(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        B, H, W, C = x.shape
        L = H * W

        x = x.reshape(B, -1, C)

        len_keep = int(L * (1 - self.mask_ratio))

        noise = mx.random.normal((B, L))

        idx_shuffle = mx.argsort(noise, axis=-1)
        idx_restore = mx.argsort(idx_shuffle, axis=-1)

        idx_keep = idx_shuffle[:, :len_keep]
        idx_keep = mx.expand_dims(idx_keep, -1)
        idx_keep = mx.repeat(idx_keep, C, axis=-1)

        x_masked = mx.take_along_axis(x, idx_keep, axis=1)
        Hr = Wr = int(len_keep**0.5)
        x_masked = x_masked.reshape(B, Hr, Wr, C)

        mask = mx.array(mx.ones((B, L)))
        mask[:, :len_keep] = 0
        mask = mx.take_along_axis(mask, idx_restore, axis=1)

        return x_masked, mask, idx_restore

    def encode(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        x = self.patch_embed(x)
        if self.pos_emb is not None:
            x = x + self.pos_emb

        x, mask, idx_restore = self.random_mask(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x)

        return x, mask, idx_restore

    def decode(self, x: mx.array, idx_restore: mx.array) -> mx.array:
        B, _, _, C = x.shape
        x = x.reshape(B, -1, C)

        mask_tokens = mx.tile(
            self.mask_token,
            (x.shape[0], int(idx_restore.shape[-1] * self.mask_ratio), 1),
        )

        x = mx.concatenate((x, mask_tokens), axis=1)
        idx_restore = mx.repeat(idx_restore[:, :, None], C, axis=-1)
        x = mx.take_along_axis(x, idx_restore, axis=1)

        L = x.shape[1]
        H = W = int(L**0.5)
        x = x.reshape(B, H, W, C)

        if self.decoder_pos_emb is not None:
            x = x + self.decoder_pos_emb

        for blk in self.decoder:
            x = blk(x)

        x = self.decoder_neck(x)

        return x

    def _prepare_loss(self, pred: mx.array, target: mx.array) -> mx.array:
        B, H, W, _ = target.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size

        target = (
            target.reshape(B, Hp, self.patch_size, Wp, self.patch_size, -1)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(B, Hp * Wp, -1)
        )

        pred = pred.reshape(B, Hp * Wp, -1)
        return target, pred

    def loss(self, pred: mx.array, target: mx.array, mask: mx.array) -> mx.array:
        target, pred = self._prepare_loss(pred, target)

        loss = (pred - target) ** 2
        loss = loss.mean(axis=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def __call__(self, x: mx.array) -> mx.array:
        latent, mask, idx_restore = self.encode(x)
        pred = self.decode(latent, idx_restore)
        loss = self.loss(pred, x, mask)
        return loss
