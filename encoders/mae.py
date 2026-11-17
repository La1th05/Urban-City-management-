"""
mae.py  --  small Masked Autoencoder for 64x64 satellite tiles.

Design (all sizes are defaults; see train_mae.py for the flags):
    tile [B, 3, 64, 64] -> 8x8 patches of 8x8 pixels -> 64 tokens
    Encoder : linear patch embedding + fixed 2D sin-cos positions + small ViT.
              Width = D from the shared contract (64), so the pooled output
              is already the [*, 64] embedding the contract asks for.
    Decoder : light ViT that sees the encoder tokens plus mask tokens and
              predicts the pixels of every patch. Used only for pretraining.
    Loss    : mean squared error on the MASKED patches only.

Embedding for downstream use:  Encoder.embed(x) -> [B, dim] (mean of all tokens,
no masking). A randomly initialised Encoder is the random-init baseline.

Run as part of the package:  from encoders.mae import MAE, Encoder
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Positional embeddings (fixed, not learned)
# ----------------------------------------------------------------------------
def sincos_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
    assert dim % 2 == 0
    omega = 1.0 / (10000 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0)))
    out = pos.float()[:, None] * omega[None, :]
    return torch.cat([out.sin(), out.cos()], dim=1)


def sincos_2d(dim: int, grid: int) -> torch.Tensor:
    """[grid*grid, dim] positions in row-major patch order."""
    assert dim % 4 == 0, "token width must be divisible by 4"
    ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    return torch.cat([sincos_1d(dim // 2, ys.reshape(-1)),
                      sincos_1d(dim // 2, xs.reshape(-1))], dim=1)


# ----------------------------------------------------------------------------
# Patch helpers
# ----------------------------------------------------------------------------
def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, (H/p)*(W/p), p*p*C], patches in row-major order."""
    b, c, h, w = x.shape
    g = h // p
    x = x.reshape(b, c, g, p, g, p).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, g * g, p * p * c)


def unpatchify(x: torch.Tensor, p: int, c: int = 3) -> torch.Tensor:
    """Inverse of patchify."""
    b, n, _ = x.shape
    g = int(round(n ** 0.5))
    x = x.reshape(b, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4)
    return x.reshape(b, c, g * p, g * p)


def random_masking(tokens: torch.Tensor, mask_ratio: float):
    """Keep a random subset of tokens per sample.

    Returns kept tokens [B, L, dim], mask [B, N] (1 = removed, 0 = kept), ids_restore [B, N].
    """
    b, n, dim = tokens.shape
    len_keep = max(1, int(n * (1.0 - mask_ratio)))
    noise = torch.rand(b, n, device=tokens.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    kept = torch.gather(tokens, 1, ids_keep[..., None].expand(-1, -1, dim))
    mask = torch.ones(b, n, device=tokens.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return kept, mask, ids_restore


def _block(dim: int, heads: int) -> nn.Module:
    return nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=4 * dim, dropout=0.0,
        activation="gelu", batch_first=True, norm_first=True)


# ----------------------------------------------------------------------------
# Encoder
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, img_size: int = 64, patch: int = 8, in_ch: int = 3,
                 dim: int = 64, depth: int = 6, heads: int = 4):
        super().__init__()
        assert img_size % patch == 0 and dim % heads == 0 and dim % 4 == 0
        self.patch, self.in_ch, self.dim = patch, in_ch, dim
        grid = img_size // patch
        self.patch_embed = nn.Linear(patch * patch * in_ch, dim)
        self.register_buffer("pos", sincos_2d(dim, grid)[None], persistent=False)
        self.blocks = nn.ModuleList([_block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0):
        tokens = self.patch_embed(patchify(x, self.patch)) + self.pos
        if mask_ratio > 0:
            tokens, mask, ids_restore = random_masking(tokens, mask_ratio)
        else:
            mask = torch.zeros(tokens.shape[:2], device=tokens.device)
            ids_restore = None
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens), mask, ids_restore

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, 64, 64] -> [B, dim]. Call model.eval() first."""
        tokens, _, _ = self(x, 0.0)
        return tokens.mean(dim=1)


# ----------------------------------------------------------------------------
# Decoder (pretraining only)
# ----------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, enc_dim: int, img_size: int = 64, patch: int = 8, in_ch: int = 3,
                 dim: int = 128, depth: int = 2, heads: int = 4):
        super().__init__()
        assert dim % heads == 0 and dim % 4 == 0
        grid = img_size // patch
        self.embed = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", sincos_2d(dim, grid)[None], persistent=False)
        self.blocks = nn.ModuleList([_block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.pred = nn.Linear(dim, patch * patch * in_ch)

    def forward(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.embed(latent)
        b, l, dim = x.shape
        n = ids_restore.shape[1]
        x = torch.cat([x, self.mask_token.expand(b, n - l, -1)], dim=1)
        x = torch.gather(x, 1, ids_restore[..., None].expand(-1, -1, dim))
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.pred(self.norm(x))


# ----------------------------------------------------------------------------
# Full MAE
# ----------------------------------------------------------------------------
class MAE(nn.Module):
    def __init__(self, img_size: int = 64, patch: int = 8, in_ch: int = 3,
                 dim: int = 64, depth: int = 6, heads: int = 4,
                 dec_dim: int = 128, dec_depth: int = 2, dec_heads: int = 4,
                 mask_ratio: float = 0.75, norm_pix_loss: bool = True):
        super().__init__()
        self.patch, self.in_ch = patch, in_ch
        self.mask_ratio, self.norm_pix_loss = mask_ratio, norm_pix_loss
        self.encoder = Encoder(img_size, patch, in_ch, dim, depth, heads)
        self.decoder = Decoder(dim, img_size, patch, in_ch, dec_dim, dec_depth, dec_heads)

    def _target(self, x: torch.Tensor):
        """Pixel targets per patch, plus the statistics needed to undo normalisation."""
        target = patchify(x, self.patch).float()
        mean = var = None
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        return target, mean, var

    def forward(self, x: torch.Tensor, mask_ratio: float | None = None):
        mr = self.mask_ratio if mask_ratio is None else mask_ratio
        latent, mask, ids_restore = self.encoder(x, mr)
        pred = self.decoder(latent, ids_restore).float()
        target, _, _ = self._target(x)
        per_patch = ((pred - target) ** 2).mean(dim=-1)
        loss = (per_patch * mask).sum() / mask.sum().clamp(min=1.0)
        return loss, pred, mask

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, mask_ratio: float | None = None):
        """Returns (masked_input, reconstruction), both [B, 3, 64, 64] in normalised pixel space."""
        _, pred, mask = self(x, mask_ratio)
        target, mean, var = self._target(x)
        raw = patchify(x, self.patch).float()
        if self.norm_pix_loss:
            pred = pred * (var + 1e-6) ** 0.5 + mean
        keep = (1.0 - mask)[..., None]
        masked = raw * keep
        recon = raw * keep + pred * mask[..., None]
        return unpatchify(masked, self.patch, self.in_ch), unpatchify(recon, self.patch, self.in_ch)