"""
MM-DiT (SD3-style multi-modal DiT) and a hierarchical variant.

The block is a true dual-stream MM-DiT: text and image tokens keep separate
qkv/mlp/modulation weights and mix through a single joint attention over the
concatenated sequence. The previous implementation mean-pooled the text into
the adaLN vector (text never entered the token sequence) and ran attention
and MLP in parallel off one norm, which roughly halved the effective depth -
it was a text-modulated DiT, not an MM-DiT.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Sequence
import einops
from flax.typing import Dtype, PrecisionLike

from .dit_common import (
    PatchSequenceEmbed, ConditioningEmbed, PatchSequenceOutput,
    neutralized_rope_freqs,
)
from .vit_common import RotaryEmbedding, AdaLNParams, apply_rotary_embedding


class MMDiTBlock(nn.Module):
    """Dual-stream MM-DiT block: per-modality weights, joint attention.

    Both streams are modulated adaLN-Zero style from the same conditioning
    vector but with separate parameters, then attend jointly over
    concat([txt, img]) and go through separate sequential MLPs.
    """
    features: int
    num_heads: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    qk_norm: bool = False

    def setup(self):
        hidden_features = int(self.features * self.mlp_ratio)
        dim_head = self.features // self.num_heads
        qkv = lambda name: nn.DenseGeneral(
            features=[self.num_heads, dim_head], axis=-1,
            dtype=self.dtype, precision=self.precision, use_bias=True, name=name)
        out_proj = lambda name: nn.DenseGeneral(
            self.features, axis=(-2, -1),
            dtype=self.dtype, precision=self.precision, name=name)
        mlp = lambda name: nn.Sequential([
            nn.Dense(features=hidden_features, dtype=self.dtype, precision=self.precision),
            nn.gelu,
            nn.Dense(features=self.features, dtype=self.dtype, precision=self.precision),
        ], name=name)
        norm = lambda name: nn.LayerNorm(
            epsilon=self.norm_epsilon, use_scale=False, use_bias=False,
            dtype=self.dtype, name=name)

        # image stream
        self.img_ada = AdaLNParams(self.features, dtype=self.dtype, precision=self.precision)
        self.img_norm1, self.img_norm2 = norm("img_norm1"), norm("img_norm2")
        self.img_q, self.img_k, self.img_v = qkv("img_to_q"), qkv("img_to_k"), qkv("img_to_v")
        self.img_out = out_proj("img_out")
        self.img_mlp = mlp("img_mlp")

        # text stream
        self.txt_ada = AdaLNParams(self.features, dtype=self.dtype, precision=self.precision)
        self.txt_norm1, self.txt_norm2 = norm("txt_norm1"), norm("txt_norm2")
        self.txt_q, self.txt_k, self.txt_v = qkv("txt_to_q"), qkv("txt_to_k"), qkv("txt_to_v")
        self.txt_out = out_proj("txt_out")
        self.txt_mlp = mlp("txt_mlp")

        if self.qk_norm:
            self.img_q_norm = nn.RMSNorm(dtype=self.dtype, name="img_q_norm")
            self.img_k_norm = nn.RMSNorm(dtype=self.dtype, name="img_k_norm")
            self.txt_q_norm = nn.RMSNorm(dtype=self.dtype, name="txt_q_norm")
            self.txt_k_norm = nn.RMSNorm(dtype=self.dtype, name="txt_k_norm")

        self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, img, txt, conditioning, freqs_cis, train: bool = False):
        S_txt = txt.shape[1]
        i_scale_mlp, i_shift_mlp, i_gate_mlp, i_scale_attn, i_shift_attn, i_gate_attn = jnp.split(
            self.img_ada(conditioning), 6, axis=-1)
        t_scale_mlp, t_shift_mlp, t_gate_mlp, t_scale_attn, t_shift_attn, t_gate_attn = jnp.split(
            self.txt_ada(conditioning), 6, axis=-1)

        # --- Joint attention ---
        img_h = self.img_norm1(img) * (1 + i_scale_attn) + i_shift_attn
        txt_h = self.txt_norm1(txt) * (1 + t_scale_attn) + t_shift_attn

        q_i, k_i, v_i = self.img_q(img_h), self.img_k(img_h), self.img_v(img_h)
        q_t, k_t, v_t = self.txt_q(txt_h), self.txt_k(txt_h), self.txt_v(txt_h)
        if self.qk_norm:
            q_i, k_i = self.img_q_norm(q_i), self.img_k_norm(k_i)
            q_t, k_t = self.txt_q_norm(q_t), self.txt_k_norm(k_t)

        # RoPE carries 2D position for image tokens only
        if freqs_cis is not None:
            freqs_cos, freqs_sin = freqs_cis
            q_i = einops.rearrange(q_i, 'b s h d -> b h s d')
            k_i = einops.rearrange(k_i, 'b s h d -> b h s d')
            q_i = apply_rotary_embedding(q_i, freqs_cos, freqs_sin)
            k_i = apply_rotary_embedding(k_i, freqs_cos, freqs_sin)
            q_i = einops.rearrange(q_i, 'b h s d -> b s h d')
            k_i = einops.rearrange(k_i, 'b h s d -> b s h d')

        q = jnp.concatenate([q_t, q_i], axis=1)
        k = jnp.concatenate([k_t, k_i], axis=1)
        v = jnp.concatenate([v_t, v_i], axis=1)

        attn = nn.dot_product_attention(
            q, k, v, dtype=self.dtype, broadcast_dropout=False, dropout_rng=None,
            precision=self.precision, force_fp32_for_softmax=self.force_fp32_for_softmax,
            deterministic=True)
        txt_attn, img_attn = attn[:, :S_txt], attn[:, S_txt:]

        img_attn = self.dropout(self.img_out(img_attn), deterministic=not train)
        txt_attn = self.dropout(self.txt_out(txt_attn), deterministic=not train)
        img = img + i_gate_attn * img_attn
        txt = txt + t_gate_attn * txt_attn

        # --- Sequential MLPs ---
        img_h = self.img_norm2(img) * (1 + i_scale_mlp) + i_shift_mlp
        img = img + i_gate_mlp * self.dropout(self.img_mlp(img_h), deterministic=not train)
        txt_h = self.txt_norm2(txt) * (1 + t_scale_mlp) + t_shift_mlp
        txt = txt + t_gate_mlp * self.dropout(self.txt_mlp(txt_h), deterministic=not train)

        return img, txt


class SimpleMMDiT(nn.Module):
    """SD3-style MM-DiT: a plain stack of dual-stream blocks."""
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    dropout_rate: float = 0.0  # Typically 0 for diffusion
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    learn_sigma: bool = False
    qk_norm: bool = False
    use_hilbert: bool = False
    use_zigzag: bool = False

    @property
    def scan_order(self):
        assert not (self.use_hilbert and self.use_zigzag), \
            "use_hilbert and use_zigzag are mutually exclusive"
        return 'hilbert' if self.use_hilbert else 'zigzag' if self.use_zigzag else 'raster'

    def setup(self):
        self.embed = PatchSequenceEmbed(
            patch_size=self.patch_size,
            emb_features=self.emb_features,
            scan_order=self.scan_order,
            dtype=self.dtype,
            precision=self.precision,
        )
        self.conditioning = ConditioningEmbed(
            emb_features=self.emb_features,
            mlp_ratio=self.mlp_ratio,
            dtype=self.dtype,
            precision=self.precision,
        )
        # text tokens enter the sequence, so they need their own projection
        self.txt_embed = nn.Dense(
            features=self.emb_features, dtype=self.dtype,
            precision=self.precision, name="txt_embed")
        self.rope = RotaryEmbedding(
            dim=self.emb_features // self.num_heads, max_seq_len=4096, dtype=self.dtype)
        self.blocks = [
            MMDiTBlock(
                features=self.emb_features,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                precision=self.precision,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                norm_epsilon=self.norm_epsilon,
                qk_norm=self.qk_norm,
                name=f"mmdit_block_{i}"
            ) for i in range(self.num_layers)
        ]
        self.output = PatchSequenceOutput(
            patch_size=self.patch_size,
            output_channels=self.output_channels,
            learn_sigma=self.learn_sigma,
            modulated=True,
            norm_epsilon=self.norm_epsilon,
            dtype=self.dtype,
            precision=self.precision,
        )

    @nn.compact
    def __call__(self, x, temb, textcontext, train: bool = False):  # textcontext is required
        assert textcontext is not None, "textcontext must be provided for SimpleMMDiT"
        B, H, W, C = x.shape

        img, inv_idx = self.embed(x)
        txt = self.txt_embed(textcontext)
        cond_emb = self.conditioning(temb, textcontext)
        freqs_cis = neutralized_rope_freqs(self.rope, img.shape[1], self.scan_order)

        for block in self.blocks:
            img, txt = block(img, txt, conditioning=cond_emb, freqs_cis=freqs_cis, train=train)

        return self.output(img, inv_idx, H, W, conditioning=cond_emb)


class PatchMerging(nn.Module):
    """
    Merges a group of patches into a single patch with increased feature dimensions.
    Used in the hierarchical structure to reduce spatial resolution and increase channels.
    """
    out_features: int
    merge_size: int = 2  # Default 2x2 patch merging
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    norm_epsilon: float = 1e-5  # Add norm for stability like in Swin Transformer

    @nn.compact
    def __call__(self, x, H_patches, W_patches):
        # x shape: [B, H*W, C]
        B, L, C = x.shape
        assert L == H_patches * \
            W_patches, f"Input length {L} doesn't match {H_patches}*{W_patches}"
        assert H_patches % self.merge_size == 0 and W_patches % self.merge_size == 0, f"Patch dimensions ({H_patches}, {W_patches}) not divisible by merge size {self.merge_size}"

        # Reshape to [B, H, W, C]
        x = x.reshape(B, H_patches, W_patches, C)

        # Merge patches - rearrange to group nearby patches
        merged = einops.rearrange(
            x,
            'b (h p1) (w p2) c -> b h w (p1 p2 c)',
            p1=self.merge_size, p2=self.merge_size
        )

        # Apply LayerNorm before projection (common practice)
        norm = nn.LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")
        merged = norm(merged) # Apply norm on [B, H/p, W/p, p*p*C]

        # Project to new dimension
        merged = nn.Dense(
            features=self.out_features,
            dtype=self.dtype,
            precision=self.precision,
            name="projection"
        )(merged)

        # Flatten back to sequence
        new_H = H_patches // self.merge_size
        new_W = W_patches // self.merge_size
        merged = merged.reshape(B, new_H * new_W, self.out_features)

        return merged, new_H, new_W

class PatchExpanding(nn.Module):
    """
    Expands patches to increase spatial resolution.
    Used in the hierarchical structure decoder path.
    """
    out_features: int
    expand_size: int = 2  # Default 2x2 patch expansion
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    norm_epsilon: float = 1e-5 # Add norm for stability

    @nn.compact
    def __call__(self, x, H_patches, W_patches):
        # x shape: [B, H*W, C]
        B, L, C = x.shape
        assert L == H_patches * W_patches, f"Input length {L} doesn't match {H_patches}*{W_patches}"

        # Project to expanded dimension first
        expanded_features = self.expand_size * self.expand_size * self.out_features
        x = nn.Dense(
            features=expanded_features,
            dtype=self.dtype,
            precision=self.precision,
            name="projection"
        )(x) # Shape [B, L, P*P*C_out]

        # Apply LayerNorm after projection
        norm = nn.LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")
        x = norm(x)

        # Reshape to spatial grid before rearranging
        x = x.reshape(B, H_patches, W_patches, expanded_features)

        # Rearrange to expand spatial dims
        expanded = einops.rearrange(
            x,
            'b h w (p1 p2 c) -> b (h p1) (w p2) c',
            p1=self.expand_size, p2=self.expand_size, c=self.out_features
        )

        # Flatten back to sequence
        new_H = H_patches * self.expand_size
        new_W = W_patches * self.expand_size
        expanded = expanded.reshape(B, new_H * new_W, self.out_features)

        return expanded, new_H, new_W


class HierarchicalMMDiT(nn.Module):
    """U-shaped MM-DiT: dual-stream blocks per stage with patch merging on the
    way down and expansion + skip fusion on the way up.

    Raster order only - the merge/expand grid reshapes assume row-major token
    order, so a hilbert scan would scramble the neighborhoods being merged.
    """
    output_channels: int = 3
    base_patch_size: int = 8  # Patch size at the *finest* resolution level (stage 0)
    emb_features: Sequence[int] = (512, 768, 1024)  # Feature dims for stages, fine to coarse
    num_layers: Sequence[int] = (4, 4, 14)  # Layers per stage, fine to coarse
    num_heads: Sequence[int] = (8, 12, 16)  # Heads per stage, fine to coarse
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    learn_sigma: bool = False
    qk_norm: bool = False

    def setup(self):
        assert len(self.emb_features) == len(self.num_layers) == len(self.num_heads), \
            "Feature dimensions, layers, and heads must have the same number of stages"
        num_stages = len(self.emb_features)

        self.embed = PatchSequenceEmbed(
            patch_size=self.base_patch_size,
            emb_features=self.emb_features[0],
            scan_order='raster',
            dtype=self.dtype,
            precision=self.precision,
        )
        # Base conditioning at the finest dim, projected per stage
        self.conditioning = ConditioningEmbed(
            emb_features=self.emb_features[0],
            mlp_ratio=self.mlp_ratio,
            dtype=self.dtype,
            precision=self.precision,
        )
        self.cond_projs = [
            nn.Dense(features=self.emb_features[i], dtype=self.dtype,
                     precision=self.precision, name=f"cond_proj_stage{i}")
            for i in range(num_stages)
        ]
        # Per-stage text streams (dims differ per stage)
        self.txt_embeds = [
            nn.Dense(features=self.emb_features[i], dtype=self.dtype,
                     precision=self.precision, name=f"txt_embed_stage{i}")
            for i in range(num_stages)
        ]
        self.ropes = [
            RotaryEmbedding(
                dim=self.emb_features[i] // self.num_heads[i],
                max_seq_len=4096, dtype=self.dtype, name=f"rope_stage_{i}")
            for i in range(num_stages)
        ]

        def stage_blocks(stage, prefix):
            return [
                MMDiTBlock(
                    features=self.emb_features[stage],
                    num_heads=self.num_heads[stage],
                    mlp_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype,
                    precision=self.precision,
                    force_fp32_for_softmax=self.force_fp32_for_softmax,
                    norm_epsilon=self.norm_epsilon,
                    qk_norm=self.qk_norm,
                    name=f"{prefix}_block_stage{stage}_{i}"
                ) for i in range(self.num_layers[stage])
            ]

        # --- Encoder path (fine to coarse) ---
        self.encoder_blocks = [stage_blocks(s, "encoder") for s in range(num_stages)]
        self.patch_mergers = [
            PatchMerging(
                out_features=self.emb_features[s + 1],
                dtype=self.dtype,
                precision=self.precision,
                norm_epsilon=self.norm_epsilon,
                name=f"patch_merger_{s}"
            ) for s in range(num_stages - 1)
        ]

        # --- Decoder path (coarse to fine), ordered for stages N-2, ..., 0 ---
        decoder_stages = list(range(num_stages - 2, -1, -1))
        self.patch_expanders = [
            PatchExpanding(
                out_features=self.emb_features[s],
                dtype=self.dtype,
                precision=self.precision,
                norm_epsilon=self.norm_epsilon,
                name=f"patch_expander_{s}"
            ) for s in decoder_stages
        ]
        self.fusion_layers = [
            nn.Sequential([
                nn.LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype),
                nn.Dense(features=self.emb_features[s], dtype=self.dtype,
                         precision=self.precision),
            ], name=f"fusion_{s}") for s in decoder_stages
        ]
        self.decoder_blocks = [stage_blocks(s, "decoder") for s in decoder_stages]

        self.output = PatchSequenceOutput(
            patch_size=self.base_patch_size,
            output_channels=self.output_channels,
            learn_sigma=self.learn_sigma,
            modulated=True,
            norm_epsilon=self.norm_epsilon,
            dtype=self.dtype,
            precision=self.precision,
        )

    @nn.compact
    def __call__(self, x, temb, textcontext, train: bool = False):
        assert textcontext is not None, "textcontext must be provided"
        B, H, W, C = x.shape
        num_stages = len(self.emb_features)
        assert H % (self.base_patch_size * (2**(num_stages - 1))) == 0 and \
               W % (self.base_patch_size * (2**(num_stages - 1))) == 0, \
            f"Image dimensions ({H},{W}) must be divisible by effective coarsest patch size {self.base_patch_size * (2**(num_stages - 1))}"

        img, _ = self.embed(x)
        cond_base = self.conditioning(temb, textcontext)
        conds = [proj(cond_base) for proj in self.cond_projs]
        txts = [embed(textcontext) for embed in self.txt_embeds]

        # --- Encoder path ---
        H_P, W_P = H // self.base_patch_size, W // self.base_patch_size
        skips = {}
        for stage in range(num_stages):
            freqs_cis = self.ropes[stage](seq_len=img.shape[1])
            txt = txts[stage]
            for block in self.encoder_blocks[stage]:
                img, txt = block(img, txt, conditioning=conds[stage], freqs_cis=freqs_cis, train=train)
            skips[stage] = img
            if stage < num_stages - 1:
                img, H_P, W_P = self.patch_mergers[stage](img, H_P, W_P)

        # --- Decoder path ---
        for i, stage in enumerate(range(num_stages - 2, -1, -1)):
            img, H_P, W_P = self.patch_expanders[i](img, H_P, W_P)
            img = self.fusion_layers[i](jnp.concatenate([img, skips[stage]], axis=-1))
            freqs_cis = self.ropes[stage](seq_len=img.shape[1])
            txt = txts[stage]
            for block in self.decoder_blocks[i]:
                img, txt = block(img, txt, conditioning=conds[stage], freqs_cis=freqs_cis, train=train)

        return self.output(img, None, H, W, conditioning=conds[0])
