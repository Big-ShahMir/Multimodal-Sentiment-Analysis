#!/usr/bin/env python3
"""
PyTorch Lightning training wrapper for AVTCAModel.

Features
--------
1. Imbalance-aware weighted regression loss (Weighted Huber).
2. SOTA-oriented validation/testing metrics with torchmetrics:
   - MAE
   - PCC
   - Thresholded macro F1 (multi-label presence proxy)
   - Thresholded macro UA (macro multilabel accuracy)
3. AdamW optimizer + linear warmup + cosine annealing scheduler.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

try:
    from lightning.pytorch import LightningModule
except Exception:
    try:
        from pytorch_lightning import LightningModule  # type: ignore[no-redef]
    except Exception:
        LightningModule = nn.Module  # type: ignore[assignment,misc]

from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score
from torchmetrics.regression import MeanAbsoluteError, PearsonCorrCoef

try:
    from .dataset import TARGET_COLUMNS
    from .data_loader import BatchDict
    from .model import AVTCAModel, AVTCAModelConfig
except (ImportError, ValueError):
    try:
        from data.dataset import TARGET_COLUMNS  # type: ignore[no-redef]
        from data.data_loader import BatchDict  # type: ignore[no-redef]
        from .model import AVTCAModel, AVTCAModelConfig  # type: ignore[no-redef]
    except ImportError:
        from data.dataset import TARGET_COLUMNS  # type: ignore[no-redef]
        from data.data_loader import BatchDict  # type: ignore[no-redef]
        from model.model import AVTCAModel, AVTCAModelConfig  # type: ignore[no-redef]


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass(frozen=True)
class SchedulerConfig:
    # If warmup_steps is None, it is derived from warmup_ratio * total_steps.
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.1
    warmup_start_factor: float = 0.1
    eta_min: float = 1e-6


class WeightedHuberLoss(nn.Module):
    """
    Weighted Huber loss for 6 continuous emotion intensities.

    The class weights are injected as a [C] vector and directly reweight each
    target dimension. This is where minority classes receive stronger penalties.
    """

    def __init__(self, class_weights: Tensor, delta: float = 1.0) -> None:
        super().__init__()
        if class_weights.ndim != 1:
            raise ValueError(f"class_weights must be 1D, got shape={tuple(class_weights.shape)}")
        if delta <= 0:
            raise ValueError("delta must be positive for Huber loss.")
        self.delta = float(delta)
        self.register_buffer("class_weights", class_weights.to(torch.float32))

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        if preds.shape != targets.shape:
            raise ValueError(f"preds and targets shapes must match, got {preds.shape} vs {targets.shape}")
        if preds.ndim != 2:
            raise ValueError(f"Expected [B, C] tensors, got shape={tuple(preds.shape)}")
        if preds.shape[1] != self.class_weights.numel():
            raise ValueError(
                f"class_weights length ({self.class_weights.numel()}) must match num classes ({preds.shape[1]})."
            )

        error = preds - targets
        abs_error = error.abs()

        quadratic = 0.5 * (error ** 2)
        linear = self.delta * (abs_error - 0.5 * self.delta)
        huber = torch.where(abs_error <= self.delta, quadratic, linear)  # [B, C]

        # Core class-imbalance injection:
        # multiply per-emotion regression error by class_weights.
        weighted = huber * self.class_weights.unsqueeze(0)
        loss = weighted.sum(dim=1) / self.class_weights.sum().clamp(min=1e-8)
        return loss.mean()


class MERLightningModule(LightningModule):
    """
    Lightning training wrapper for AVTCAModel.

    Parameters
    ----------
    model:
        Instantiated AVTCAModel. If None, `model_config` must be provided.
    model_config:
        Configuration used to construct AVTCAModel when `model` is None.
    class_weights:
        Tensor of shape [6] for imbalance-aware regression weighting.
    optimizer_config:
        AdamW hyperparameters.
    scheduler_config:
        Warmup + cosine scheduler settings.
    regression_loss:
        "huber" or "mse" for weighted regression.
    huber_delta:
        Delta parameter for weighted Huber.
    presence_threshold:
        Intensity threshold used for presence-based metrics (F1/UA).
    target_columns:
        Label order for consistency with class weights and split stats.
    metric_clamp_range:
        Clamp predictions to this range before metric computation.
    sync_dist:
        Set True for DDP-safe logging aggregation.
    """

    def __init__(
        self,
        model: Optional[AVTCAModel] = None,
        model_config: Optional[AVTCAModelConfig] = None,
        class_weights: Optional[Tensor] = None,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        scheduler_config: SchedulerConfig = SchedulerConfig(),
        regression_loss: Literal["huber", "mse"] = "huber",
        huber_delta: float = 1.0,
        presence_threshold: float = 0.1,
        target_columns: Sequence[str] = TARGET_COLUMNS,
        metric_clamp_range: Tuple[float, float] = (0.0, 3.0),
        sync_dist: bool = True,
    ) -> None:
        super().__init__()
        if model is None and model_config is None:
            raise ValueError("Either `model` or `model_config` must be provided.")
        if model is None:
            assert model_config is not None
            model = AVTCAModel(model_config)

        self.model = model
        self.target_columns = list(target_columns)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.regression_loss_name = regression_loss
        self.huber_delta = float(huber_delta)
        self.presence_threshold = float(presence_threshold)
        self.metric_clamp_min, self.metric_clamp_max = metric_clamp_range
        self.sync_dist_enabled = bool(sync_dist)

        output_dim = self.model.config.num_outputs
        if class_weights is None:
            class_weights = torch.ones(output_dim, dtype=torch.float32)
        if class_weights.numel() != output_dim:
            raise ValueError(
                f"class_weights must have {output_dim} elements to match model outputs, "
                f"got {class_weights.numel()}."
            )

        self.register_buffer("class_weights", class_weights.to(torch.float32))

        if regression_loss == "huber":
            self.loss_fn: nn.Module = WeightedHuberLoss(class_weights=self.class_weights, delta=self.huber_delta)
        elif regression_loss == "mse":
            self.loss_fn = self._build_weighted_mse_loss()
        else:
            raise ValueError(f"Unsupported regression_loss={regression_loss}")

        # Validation metrics.
        self.val_mae = MeanAbsoluteError()
        self.val_pcc = PearsonCorrCoef()
        self.val_macro_f1 = MultilabelF1Score(
            num_labels=output_dim,
            threshold=self._presence_prob_threshold(),
            average="macro",
        )
        self.val_macro_ua = MultilabelAccuracy(
            num_labels=output_dim,
            threshold=self._presence_prob_threshold(),
            average="macro",
        )

        # Test metrics.
        self.test_mae = MeanAbsoluteError()
        self.test_pcc = PearsonCorrCoef()
        self.test_macro_f1 = MultilabelF1Score(
            num_labels=output_dim,
            threshold=self._presence_prob_threshold(),
            average="macro",
        )
        self.test_macro_ua = MultilabelAccuracy(
            num_labels=output_dim,
            threshold=self._presence_prob_threshold(),
            average="macro",
        )

        self.save_hyperparameters(
            ignore=[
                "model",
                "class_weights",
                "loss_fn",
                "val_mae",
                "val_pcc",
                "val_macro_f1",
                "val_macro_ua",
                "test_mae",
                "test_pcc",
                "test_macro_f1",
                "test_macro_ua",
            ]
        )

    def _build_weighted_mse_loss(self) -> nn.Module:
        class_weights = self.class_weights

        class _WeightedMSE(nn.Module):
            def __init__(self, cw: Tensor) -> None:
                super().__init__()
                self.register_buffer("cw", cw)

            def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
                sq = (preds - targets) ** 2
                weighted = sq * self.cw.unsqueeze(0)
                per_sample = weighted.sum(dim=1) / self.cw.sum().clamp(min=1e-8)
                return per_sample.mean()

        return _WeightedMSE(class_weights)

    def _presence_prob_threshold(self) -> float:
        # Metrics consume probabilities in [0, 1], while intensity targets are in [0, 3].
        return float(self.presence_threshold / max(self.metric_clamp_max, 1e-8))

    def _intensity_to_presence_probs(self, preds: Tensor) -> Tensor:
        # Convert intensity predictions into [0,1] presence proxy for thresholded multilabel metrics.
        clipped = preds.clamp(min=self.metric_clamp_min, max=self.metric_clamp_max)
        return clipped / max(self.metric_clamp_max, 1e-8)

    def _targets_to_presence(self, targets: Tensor) -> Tensor:
        return (targets > self.presence_threshold).to(torch.int)

    @staticmethod
    def _extract_batch_tensors(batch: BatchDict) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            batch["audio_features"],
            batch["audio_attention_mask"],
            batch["video_features"],
            batch["video_attention_mask"],
            batch["labels"],
        )

    @staticmethod
    def _safe_metric_value(value: Tensor) -> Tensor:
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        audio_features: Tensor,
        audio_attention_mask: Tensor,
        video_features: Tensor,
        video_attention_mask: Tensor,
    ) -> Tensor:
        return self.model(
            audio_features=audio_features,
            audio_attention_mask=audio_attention_mask,
            video_features=video_features,
            video_attention_mask=video_attention_mask,
        )

    def training_step(self, batch: BatchDict, batch_idx: int) -> Tensor:
        audio_x, audio_m, video_x, video_m, targets = self._extract_batch_tensors(batch)
        preds = self(
            audio_features=audio_x,
            audio_attention_mask=audio_m,
            video_features=video_x,
            video_attention_mask=video_m,
        )
        loss = self.loss_fn(preds, targets)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=targets.size(0),
            sync_dist=self.sync_dist_enabled,
        )
        return loss

    def validation_step(self, batch: BatchDict, batch_idx: int) -> Tensor:
        audio_x, audio_m, video_x, video_m, targets = self._extract_batch_tensors(batch)
        preds = self(
            audio_features=audio_x,
            audio_attention_mask=audio_m,
            video_features=video_x,
            video_attention_mask=video_m,
        )
        loss = self.loss_fn(preds, targets)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=targets.size(0),
            sync_dist=self.sync_dist_enabled,
        )

        metric_preds = preds.clamp(min=self.metric_clamp_min, max=self.metric_clamp_max)
        self.val_mae.update(metric_preds, targets)
        self.val_pcc.update(metric_preds.reshape(-1), targets.reshape(-1))

        presence_probs = self._intensity_to_presence_probs(metric_preds)
        presence_targets = self._targets_to_presence(targets)
        self.val_macro_f1.update(presence_probs, presence_targets)
        self.val_macro_ua.update(presence_probs, presence_targets)
        return loss

    def test_step(self, batch: BatchDict, batch_idx: int) -> Tensor:
        audio_x, audio_m, video_x, video_m, targets = self._extract_batch_tensors(batch)
        preds = self(
            audio_features=audio_x,
            audio_attention_mask=audio_m,
            video_features=video_x,
            video_attention_mask=video_m,
        )
        loss = self.loss_fn(preds, targets)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=targets.size(0),
            sync_dist=self.sync_dist_enabled,
        )

        metric_preds = preds.clamp(min=self.metric_clamp_min, max=self.metric_clamp_max)
        self.test_mae.update(metric_preds, targets)
        self.test_pcc.update(metric_preds.reshape(-1), targets.reshape(-1))

        presence_probs = self._intensity_to_presence_probs(metric_preds)
        presence_targets = self._targets_to_presence(targets)
        self.test_macro_f1.update(presence_probs, presence_targets)
        self.test_macro_ua.update(presence_probs, presence_targets)
        return loss

    def on_validation_epoch_end(self) -> None:
        val_mae = self._safe_metric_value(self.val_mae.compute())
        val_pcc = self._safe_metric_value(self.val_pcc.compute())
        val_f1 = self._safe_metric_value(self.val_macro_f1.compute())
        val_ua = self._safe_metric_value(self.val_macro_ua.compute())

        self.log("val_mae", val_mae, prog_bar=True, logger=True, sync_dist=self.sync_dist_enabled)
        self.log("val_pcc", val_pcc, prog_bar=True, logger=True, sync_dist=self.sync_dist_enabled)
        self.log(
            "val_macro_f1",
            val_f1,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist_enabled,
        )
        self.log(
            "val_macro_ua",
            val_ua,
            prog_bar=False,
            logger=True,
            sync_dist=self.sync_dist_enabled,
        )
        self.val_mae.reset()
        self.val_pcc.reset()
        self.val_macro_f1.reset()
        self.val_macro_ua.reset()

    def on_test_epoch_end(self) -> None:
        test_mae = self._safe_metric_value(self.test_mae.compute())
        test_pcc = self._safe_metric_value(self.test_pcc.compute())
        test_f1 = self._safe_metric_value(self.test_macro_f1.compute())
        test_ua = self._safe_metric_value(self.test_macro_ua.compute())

        self.log("test_mae", test_mae, prog_bar=True, logger=True, sync_dist=self.sync_dist_enabled)
        self.log("test_pcc", test_pcc, prog_bar=True, logger=True, sync_dist=self.sync_dist_enabled)
        self.log(
            "test_macro_f1",
            test_f1,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_dist_enabled,
        )
        self.log(
            "test_macro_ua",
            test_ua,
            prog_bar=False,
            logger=True,
            sync_dist=self.sync_dist_enabled,
        )
        self.test_mae.reset()
        self.test_pcc.reset()
        self.test_macro_f1.reset()
        self.test_macro_ua.reset()

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = AdamW(
            self.parameters(),
            lr=self.optimizer_config.lr,
            betas=self.optimizer_config.betas,
            eps=self.optimizer_config.eps,
            weight_decay=self.optimizer_config.weight_decay,
        )

        total_steps = None
        if hasattr(self, "trainer") and self.trainer is not None:
            # Available when fit is initialized; used for step-level schedulers.
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)

        if total_steps is None or total_steps <= 1:
            # Safe fallback without scheduler if step estimate is unavailable.
            return optimizer

        if self.scheduler_config.warmup_steps is not None:
            warmup_steps = int(self.scheduler_config.warmup_steps)
        else:
            warmup_steps = int(total_steps * self.scheduler_config.warmup_ratio)
        warmup_steps = max(0, min(warmup_steps, total_steps - 1))

        if warmup_steps > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=self.scheduler_config.warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=self.scheduler_config.eta_min,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps),
                eta_min=self.scheduler_config.eta_min,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @classmethod
    def class_weights_from_split_stats(
        cls,
        split_stats_path: Path,
        target_columns: Sequence[str] = TARGET_COLUMNS,
        split: str = "train",
        strategy: Literal["inverse_presence", "inverse_sqrt_presence"] = "inverse_presence",
        eps: float = 1e-6,
        min_weight: float = 0.1,
        max_weight: float = 20.0,
        normalize_mean_to_one: bool = True,
    ) -> Tensor:
        """
        Build class weights from `*_split_stats.json` generated in build_manifest.py.

        Expected source key per class:
            splits[split]["emotions"][emotion]["presence_ratio_gt0"]

        This gives higher weight to rarer classes (e.g., fearful/disgust).
        """
        with Path(split_stats_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        emotions_block = payload["splits"][split]["emotions"]
        ratios = []
        for emotion in target_columns:
            ratio = float(emotions_block[emotion]["presence_ratio_gt0"])
            ratios.append(max(ratio, eps))

        ratio_tensor = torch.tensor(ratios, dtype=torch.float32)
        if strategy == "inverse_presence":
            weights = 1.0 / ratio_tensor
        else:
            weights = 1.0 / torch.sqrt(ratio_tensor)

        if normalize_mean_to_one:
            weights = weights / weights.mean().clamp(min=eps)

        weights = weights.clamp(min=min_weight, max=max_weight)
        return weights
