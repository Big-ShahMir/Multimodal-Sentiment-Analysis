#!/usr/bin/env python3
"""
DataLoader and collation utilities for the multimodal MER dataset.

This module provides:
1. A custom collator for variable-length audio/video token sequences.
2. Optimized DataLoader factory functions for train/val/test splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, TypedDict

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

try:
    from .dataset import (
        AudioFoundationExtractor,
        AudioExtractorConfig,
        DatasetSample,
        MOSEIPklDataset,
        MultimodalManifestDataset,
        TARGET_COLUMNS,
        VideoExtractorConfig,
        VideoFoundationExtractor,
    )
except ImportError:
    from dataset import (  # type: ignore[no-redef]
        AudioFoundationExtractor,
        AudioExtractorConfig,
        DatasetSample,
        MOSEIPklDataset,
        MultimodalManifestDataset,
        TARGET_COLUMNS,
        VideoExtractorConfig,
        VideoFoundationExtractor,
    )


class BatchDict(TypedDict):
    video_ids: List[str]
    utterance_ids: List[str]
    speaker_ids: List[str]
    audio_features: Tensor
    audio_attention_mask: Tensor
    video_features: Tensor
    video_attention_mask: Tensor
    labels: Tensor


@dataclass(frozen=True)
class CollateConfig:
    max_audio_tokens: Optional[int] = None
    max_video_tokens: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 2


def _truncate_if_needed(sequence: Tensor, max_len: Optional[int]) -> Tensor:
    if max_len is None:
        return sequence
    return sequence[:max_len]


def _compute_pad_length(max_len: int, pad_to_multiple_of: Optional[int]) -> int:
    if pad_to_multiple_of is None or pad_to_multiple_of <= 1:
        return max_len
    remainder = max_len % pad_to_multiple_of
    if remainder == 0:
        return max_len
    return max_len + (pad_to_multiple_of - remainder)


def _build_attention_mask(lengths: Tensor, max_len: int) -> Tensor:
    positions = torch.arange(max_len, dtype=torch.long).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)
    return mask


class MultimodalCollator:
    """
    Custom collator for variable-length multimodal sequences.

    Outputs:
        audio_features:       [B, T_audio_max, D_audio]
        audio_attention_mask: [B, T_audio_max] (True for valid tokens)
        video_features:       [B, T_video_max, D_video]
        video_attention_mask: [B, T_video_max] (True for valid tokens)
        labels:               [B, 6]
    """

    def __init__(self, config: Optional[CollateConfig] = None) -> None:
        self.config = config or CollateConfig()

    def __call__(self, batch: Sequence[DatasetSample]) -> BatchDict:
        if len(batch) == 0:
            raise ValueError("Received empty batch in collate_fn.")

        video_ids = [sample["video_id"] for sample in batch]
        utterance_ids = [sample["utterance_id"] for sample in batch]
        speaker_ids = [sample["speaker_id"] for sample in batch]

        audio_seqs = [
            _truncate_if_needed(sample["audio_features"], self.config.max_audio_tokens)
            for sample in batch
        ]
        video_seqs = [
            _truncate_if_needed(sample["video_features"], self.config.max_video_tokens)
            for sample in batch
        ]

        labels = torch.stack([sample["labels"] for sample in batch], dim=0).to(torch.float32)

        audio_lengths = torch.tensor([seq.shape[0] for seq in audio_seqs], dtype=torch.long)
        video_lengths = torch.tensor([seq.shape[0] for seq in video_seqs], dtype=torch.long)

        audio_padded = pad_sequence(audio_seqs, batch_first=True)
        video_padded = pad_sequence(video_seqs, batch_first=True)

        target_audio_len = _compute_pad_length(audio_padded.shape[1], self.config.pad_to_multiple_of)
        target_video_len = _compute_pad_length(video_padded.shape[1], self.config.pad_to_multiple_of)

        if target_audio_len > audio_padded.shape[1]:
            pad_size = target_audio_len - audio_padded.shape[1]
            audio_padded = torch.nn.functional.pad(audio_padded, (0, 0, 0, pad_size))

        if target_video_len > video_padded.shape[1]:
            pad_size = target_video_len - video_padded.shape[1]
            video_padded = torch.nn.functional.pad(video_padded, (0, 0, 0, pad_size))

        audio_mask = _build_attention_mask(audio_lengths, audio_padded.shape[1])
        video_mask = _build_attention_mask(video_lengths, video_padded.shape[1])

        return BatchDict(
            video_ids=video_ids,
            utterance_ids=utterance_ids,
            speaker_ids=speaker_ids,
            audio_features=audio_padded,
            audio_attention_mask=audio_mask,
            video_features=video_padded,
            video_attention_mask=video_mask,
            labels=labels,
        )


def build_dataloader(
    dataset: MultimodalManifestDataset,
    dataloader_config: DataLoaderConfig,
    collate_config: Optional[CollateConfig] = None,
) -> DataLoader[BatchDict]:
    """
    Build a DataLoader with standard performance settings.

    Important:
    - If either backbone is set to fine-tune mode (`freeze_backbone=False`),
      use `num_workers=0` to avoid multiprocessing/autograd conflicts.
    """
    fine_tune_mode = False
    if hasattr(dataset, "audio_extractor") and hasattr(dataset, "video_extractor"):
        fine_tune_mode = (
            (not dataset.audio_extractor.freeze_backbone)
            or (not dataset.video_extractor.freeze_backbone)
        )
    if fine_tune_mode and dataloader_config.num_workers > 0:
        raise ValueError(
            "Fine-tuning backbones inside Dataset requires num_workers=0. "
            "Set freeze_backbone=True for multiprocessing feature extraction."
        )

    collate_fn = MultimodalCollator(collate_config)
    use_persistent_workers = dataloader_config.persistent_workers and dataloader_config.num_workers > 0

    kwargs: Dict[str, object] = {
        "dataset": dataset,
        "batch_size": dataloader_config.batch_size,
        "shuffle": dataloader_config.shuffle,
        "num_workers": dataloader_config.num_workers,
        "pin_memory": dataloader_config.pin_memory,
        "drop_last": dataloader_config.drop_last,
        "persistent_workers": use_persistent_workers,
        "collate_fn": collate_fn,
    }
    if dataloader_config.num_workers > 0:
        kwargs["prefetch_factor"] = dataloader_config.prefetch_factor

    return DataLoader(**kwargs)


def build_split_dataloaders(
    manifest_path: Path,
    audio_cfg: AudioExtractorConfig,
    video_cfg: VideoExtractorConfig,
    train_loader_cfg: DataLoaderConfig,
    eval_loader_cfg: Optional[DataLoaderConfig] = None,
    collate_cfg: Optional[CollateConfig] = None,
    target_columns: Sequence[str] = TARGET_COLUMNS,
    strict_path_check: bool = True,
) -> Dict[Literal["train", "val", "test"], DataLoader[BatchDict]]:
    """
    Convenience factory for train/val/test DataLoaders.
    """
    eval_loader_cfg = eval_loader_cfg or DataLoaderConfig(
        batch_size=train_loader_cfg.batch_size,
        num_workers=train_loader_cfg.num_workers,
        pin_memory=train_loader_cfg.pin_memory,
        shuffle=False,
        drop_last=False,
        persistent_workers=train_loader_cfg.persistent_workers,
        prefetch_factor=train_loader_cfg.prefetch_factor,
    )

    # Shared extractors across splits to avoid reloading weights 3x.
    audio_extractor = AudioFoundationExtractor(audio_cfg)
    video_extractor = VideoFoundationExtractor(video_cfg)

    train_dataset = MultimodalManifestDataset(
        manifest_path=manifest_path,
        audio_extractor=audio_extractor,
        video_extractor=video_extractor,
        split="train",
        target_columns=target_columns,
        strict_path_check=strict_path_check,
    )
    val_dataset = MultimodalManifestDataset(
        manifest_path=manifest_path,
        audio_extractor=audio_extractor,
        video_extractor=video_extractor,
        split="val",
        target_columns=target_columns,
        strict_path_check=strict_path_check,
    )
    test_dataset = MultimodalManifestDataset(
        manifest_path=manifest_path,
        audio_extractor=audio_extractor,
        video_extractor=video_extractor,
        split="test",
        target_columns=target_columns,
        strict_path_check=strict_path_check,
    )

    train_loader = build_dataloader(
        dataset=train_dataset,
        dataloader_config=train_loader_cfg,
        collate_config=collate_cfg,
    )
    val_loader = build_dataloader(
        dataset=val_dataset,
        dataloader_config=eval_loader_cfg,
        collate_config=collate_cfg,
    )
    test_loader = build_dataloader(
        dataset=test_dataset,
        dataloader_config=eval_loader_cfg,
        collate_config=collate_cfg,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def build_split_dataloaders_pkl(
    pkl_path: Path,
    train_loader_cfg: DataLoaderConfig,
    eval_loader_cfg: Optional[DataLoaderConfig] = None,
    collate_cfg: Optional[CollateConfig] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 561,
) -> Dict[Literal["train", "val", "test"], DataLoader[BatchDict]]:
    """Build train/val/test DataLoaders from Zenodo processed_mosei.pkl."""
    eval_loader_cfg = eval_loader_cfg or DataLoaderConfig(
        batch_size=train_loader_cfg.batch_size,
        num_workers=train_loader_cfg.num_workers,
        pin_memory=train_loader_cfg.pin_memory,
        shuffle=False,
        drop_last=False,
        persistent_workers=train_loader_cfg.persistent_workers,
        prefetch_factor=train_loader_cfg.prefetch_factor,
    )
    collate_fn = MultimodalCollator(collate_cfg)
    pkl_path = Path(pkl_path)
    train_ds = MOSEIPklDataset(pkl_path, "train", train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    val_ds = MOSEIPklDataset(pkl_path, "val", train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    test_ds = MOSEIPklDataset(pkl_path, "test", train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    use_persistent = train_loader_cfg.persistent_workers and train_loader_cfg.num_workers > 0
    kwargs: Dict[str, object] = {
        "batch_size": train_loader_cfg.batch_size,
        "pin_memory": train_loader_cfg.pin_memory,
        "drop_last": train_loader_cfg.drop_last,
        "persistent_workers": use_persistent,
        "collate_fn": collate_fn,
    }
    if train_loader_cfg.num_workers > 0:
        kwargs["prefetch_factor"] = train_loader_cfg.prefetch_factor
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=train_loader_cfg.num_workers, **kwargs)
    kwargs["shuffle"] = False
    kwargs["drop_last"] = False
    val_loader = DataLoader(val_ds, num_workers=eval_loader_cfg.num_workers, **kwargs)
    test_loader = DataLoader(test_ds, num_workers=eval_loader_cfg.num_workers, **kwargs)
    return {"train": train_loader, "val": val_loader, "test": test_loader}
