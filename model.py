#!/usr/bin/env python3
"""
Core multimodal fusion architecture inspired by AVT-CA.

Pipeline
--------
1. Project audio/video features to shared latent dim d.
2. Bidirectional cross-attention in parallel:
   - Audio queries Video (A -> V)
   - Video queries Audio (V -> A)
3. Self-attention refinement per modality (TransformerEncoder).
4. Masked temporal pooling.
5. Fusion head -> 6 continuous logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AVTCAModelConfig:
    audio_input_dim: int
    video_input_dim: int
    latent_dim: int = 256
    num_heads: int = 8
    num_self_attn_layers: int = 2
    ff_multiplier: int = 4
    dropout: float = 0.1
    pooling: Literal["max", "attn"] = "max"
    fusion: Literal["concat", "add"] = "concat"
    num_outputs: int = 6


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequence features.
    """

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D]
        t = x.size(1)
        x = x + self.pe[:, :t, :]
        return self.dropout(x)


class MaskedAttentionPooling(nn.Module):
    """
    Learnable attention pooling over time with validity masking.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        # x: [B, T, D], valid_mask: [B, T] True for valid tokens
        logits = self.score(x).squeeze(-1)  # [B, T]
        logits = logits.masked_fill(~valid_mask, float("-inf"))
        weights = torch.softmax(logits, dim=-1)  # [B, T]
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, D]
        return pooled


def masked_max_pool(x: Tensor, valid_mask: Tensor) -> Tensor:
    """
    Masked temporal max pooling.
    """
    # x: [B, T, D], valid_mask: [B, T]
    x_masked = x.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
    pooled = x_masked.max(dim=1).values
    # Handle all-pad edge cases safely.
    pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
    return pooled


class AVTCAModel(nn.Module):
    """
    Bidirectional cross-attention audio-video fusion model.

    Inputs
    ------
    audio_features: Tensor [B, T_a, D_a]
    audio_attention_mask: Tensor [B, T_a] (True=valid token)
    video_features: Tensor [B, T_v, D_v]
    video_attention_mask: Tensor [B, T_v] (True=valid token)

    Output
    ------
    logits: Tensor [B, 6] (continuous logits for 6 target emotions)
    """

    def __init__(self, config: AVTCAModelConfig) -> None:
        super().__init__()
        self.config = config

        d = config.latent_dim
        ff_dim = config.ff_multiplier * d

        # Dimensionality alignment into shared latent space.
        self.audio_proj = nn.Sequential(
            nn.Linear(config.audio_input_dim, d),
            nn.LayerNorm(d),
            nn.Dropout(config.dropout),
        )
        self.video_proj = nn.Sequential(
            nn.Linear(config.video_input_dim, d),
            nn.LayerNorm(d),
            nn.Dropout(config.dropout),
        )

        self.pos_enc = SinusoidalPositionalEncoding(d_model=d, dropout=config.dropout)

        # Bidirectional cross-attention.
        self.cross_attn_a2v = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_attn_v2a = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_dropout = nn.Dropout(config.dropout)
        self.cross_norm_audio = nn.LayerNorm(d)
        self.cross_norm_video = nn.LayerNorm(d)

        # Self-attention refinement after cross-modal fusion.
        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_heads,
            dim_feedforward=ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        video_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_heads,
            dim_feedforward=ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.audio_refiner = nn.TransformerEncoder(
            encoder_layer=audio_encoder_layer,
            num_layers=config.num_self_attn_layers,
        )
        self.video_refiner = nn.TransformerEncoder(
            encoder_layer=video_encoder_layer,
            num_layers=config.num_self_attn_layers,
        )

        if config.pooling == "attn":
            self.audio_pool = MaskedAttentionPooling(d)
            self.video_pool = MaskedAttentionPooling(d)
        else:
            self.audio_pool = None
            self.video_pool = None

        fused_dim = 2 * d if config.fusion == "concat" else d
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ff_dim, config.num_outputs),
        )

    @staticmethod
    def _to_bool_mask(mask: Tensor, name: str) -> Tensor:
        if mask.ndim != 2:
            raise ValueError(f"{name} must have shape [B, T], got {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        return mask

    def _cross_attention(
        self,
        a: Tensor,
        a_valid: Tensor,
        v: Tensor,
        v_valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # key_padding_mask expects True at padding positions.
        a_kpm = ~a_valid
        v_kpm = ~v_valid

        # A -> V: audio queries video.
        a_from_v, _ = self.cross_attn_a2v(
            query=a,
            key=v,
            value=v,
            key_padding_mask=v_kpm,
            need_weights=False,
        )
        # V -> A: video queries audio.
        v_from_a, _ = self.cross_attn_v2a(
            query=v,
            key=a,
            value=a,
            key_padding_mask=a_kpm,
            need_weights=False,
        )

        a_fused = self.cross_norm_audio(a + self.cross_dropout(a_from_v))
        v_fused = self.cross_norm_video(v + self.cross_dropout(v_from_a))
        return a_fused, v_fused

    def _pool(self, a: Tensor, a_valid: Tensor, v: Tensor, v_valid: Tensor) -> Tuple[Tensor, Tensor]:
        if self.config.pooling == "attn":
            assert self.audio_pool is not None and self.video_pool is not None
            a_pooled = self.audio_pool(a, a_valid)
            v_pooled = self.video_pool(v, v_valid)
            return a_pooled, v_pooled

        a_pooled = masked_max_pool(a, a_valid)
        v_pooled = masked_max_pool(v, v_valid)
        return a_pooled, v_pooled

    def forward(
        self,
        audio_features: Tensor,
        audio_attention_mask: Tensor,
        video_features: Tensor,
        video_attention_mask: Tensor,
    ) -> Tensor:
        """
        Forward pass for multimodal fusion.
        """
        if audio_features.ndim != 3:
            raise ValueError(f"audio_features must be [B, T_a, D_a], got {tuple(audio_features.shape)}")
        if video_features.ndim != 3:
            raise ValueError(f"video_features must be [B, T_v, D_v], got {tuple(video_features.shape)}")

        a_valid = self._to_bool_mask(audio_attention_mask, "audio_attention_mask")
        v_valid = self._to_bool_mask(video_attention_mask, "video_attention_mask")

        # 1) Project to shared space + position encode.
        a = self.pos_enc(self.audio_proj(audio_features))
        v = self.pos_enc(self.video_proj(video_features))

        # 2) Bidirectional cross-attention.
        a_fused, v_fused = self._cross_attention(a=a, a_valid=a_valid, v=v, v_valid=v_valid)

        # 3) Self-attention refinement.
        a_refined = self.audio_refiner(a_fused, src_key_padding_mask=~a_valid)
        v_refined = self.video_refiner(v_fused, src_key_padding_mask=~v_valid)

        # 4) Temporal pooling.
        a_pooled, v_pooled = self._pool(a_refined, a_valid, v_refined, v_valid)

        # 5) Modality fusion + prediction.
        if self.config.fusion == "add":
            fused = a_pooled + v_pooled
        else:
            fused = torch.cat([a_pooled, v_pooled], dim=-1)

        logits = self.prediction_head(fused)  # [B, 6]
        return logits
