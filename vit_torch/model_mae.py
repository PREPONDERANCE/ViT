import torch

from torch import nn
from .model_vit import VisionTransformer, Block


class MaskedAutoEncoderViT(VisionTransformer):
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
        decoder_depth: int = 8,
        decoder_num_heads: int = 8,
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

        self.blocks = self._construct_block(
            depth=depth,
            emb_dim=emb_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
            input_size=img_size // (patch_size / (1 - mask_ratio)),
        )

        self.decoder = self._construct_block(
            depth=decoder_depth,
            emb_dim=out_chan,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_rel_pos=use_rel_pos,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes,
            input_size=img_size // patch_size,
        )

        self.decoder_pos_emb = None
        if use_abs_pos:
            self.decoder_pos_emb = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, out_chan)
            )

        self.decoder_neck = nn.Linear(out_chan, patch_size**2 * 3)

        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, out_chan))

    def _construct_block(
        self,
        depth: int,
        emb_dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        norm_layer: type[nn.Module],
        act_layer: type[nn.Module],
        use_rel_pos: bool,
        window_size: int,
        global_attn_indexes: tuple[int, ...],
        input_size: int,
    ) -> list[Block]:
        return [
            Block(
                emb_dim=emb_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i in global_attn_indexes else 0,
                input_size=(input_size, input_size),
            )
            for i in range(depth)
        ]

    def random_mask(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, W, C = x.shape
        L = H * W
        x = x.view(B, L, C)

        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.randn(B, L)

        idx_shuffle = torch.argsort(noise)
        idx_restore = torch.argsort(idx_shuffle)

        idx_keep = idx_shuffle[:, :len_keep]
        x = torch.gather(x, dim=1, index=idx_keep.unsqueeze(-1).repeat(1, 1, C))
        Hr = Wr = int(len_keep ** (0.5))
        x = x.view(B, Hr, Wr, C)

        mask = torch.ones(B, L)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=idx_restore)

        return x, mask, idx_restore

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_emb(x)
        if self.pos_emb is not None:
            x = x + self.pos_emb

        x, mask, idx_restore = self.random_mask(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1)

        return x, mask, idx_restore

    def decode(self, x: torch.Tensor, idx_restore: torch.Tensor) -> torch.Tensor:
        B, Hr, Wr, C = x.shape
        L = Hr * Wr
        x = x.view(B, L, C)

        original_len = int(L / (1 - self.mask_ratio))
        H = W = int(original_len ** (0.5))

        mask_tokens = self.mask_token.repeat(B, idx_restore.shape[1] - x.shape[1], 1)
        x = torch.cat((x, mask_tokens), dim=1)
        x = torch.gather(x, dim=1, index=idx_restore.unsqueeze(-1).repeat(1, 1, C))
        x = x.view(B, H, W, C)

        if self.decoder_pos_emb is not None:
            x = x + self.decoder_pos_emb

        for blk in self.decoder:
            x = blk(x)

        x = self.decoder_neck(x)
        return x

    def _prepare_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ps = self.patch_size

        B, _, _, C = pred.shape
        target = target.permute(0, 2, 3, 1)
        _, H, W, _ = target.shape
        pred = pred.view(B, -1, C)

        target = (
            target.view(B, H // ps, ps, W // ps, ps, 3)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(B, H * W // ps // ps, ps**2 * 3)
        )

        return pred, target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, mask, idx_restore = self.encode(x)
        pred = self.decode(latent, idx_restore)

        pred, target = self._prepare_loss(pred, x)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss
