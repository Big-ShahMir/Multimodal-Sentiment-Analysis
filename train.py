#!/usr/bin/env python3
"""
Main training entry point for the multimodal MER pipeline.

Responsibilities
----------------
1. Parse runtime/config arguments.
2. Optionally build manifest + split stats (media-path mode).
3. Support two training modes:
   - On-the-fly media feature extraction via Dataset/DataModule.
   - Precomputed .pt feature loading from ETL output manifests.
4. Configure logger/callbacks and launch training/testing.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
except Exception:
    from pytorch_lightning import Trainer, seed_everything  # type: ignore[no-redef]
    from pytorch_lightning.callbacks import (  # type: ignore[no-redef]
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from pytorch_lightning.loggers import TensorBoardLogger  # type: ignore[no-redef]

try:
    from .build_manifest import ManifestBuilder, PipelineConfig
    from .data_loader import BatchDict, CollateConfig, DataLoaderConfig, MultimodalCollator
    from .datamodule import MERDataModule
    from .dataset import AudioExtractorConfig, TARGET_COLUMNS, VideoExtractorConfig
    from .lightning_module import MERLightningModule, OptimizerConfig, SchedulerConfig
    from .model import AVTCAModel, AVTCAModelConfig
except ImportError:
    from build_manifest import ManifestBuilder, PipelineConfig  # type: ignore[no-redef]
    from data_loader import BatchDict, CollateConfig, DataLoaderConfig, MultimodalCollator  # type: ignore[no-redef]
    from datamodule import MERDataModule  # type: ignore[no-redef]
    from dataset import AudioExtractorConfig, TARGET_COLUMNS, VideoExtractorConfig  # type: ignore[no-redef]
    from lightning_module import MERLightningModule, OptimizerConfig, SchedulerConfig  # type: ignore[no-redef]
    from model import AVTCAModel, AVTCAModelConfig  # type: ignore[no-redef]


LOGGER = logging.getLogger("train")
PRECOMPUTED_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "video_id",
    "utterance_id",
    "audio_feature_path",
    "video_feature_path",
    *TARGET_COLUMNS,
)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AVT-CA multimodal MER model with PyTorch Lightning.")

    # Core paths.
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Root data directory.")
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=None,
        help=(
            "Path to manifest file. For media-path mode: csv/jsonl/parquet with audio/video paths. "
            "For precomputed mode: csv/jsonl/parquet with audio_feature_path/video_feature_path."
        ),
    )
    parser.add_argument(
        "--split_stats_path",
        type=Path,
        default=None,
        help="Path to split stats JSON. Defaults to <manifest_stem>_split_stats.json.",
    )
    parser.add_argument(
        "--use_precomputed_features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, train from precomputed .pt feature paths instead of raw media paths.",
    )
    parser.add_argument(
        "--precomputed_strict_path_check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, validate feature tensor paths exist before each load.",
    )

    # Optional manifest build.
    parser.add_argument(
        "--build_manifest",
        action="store_true",
        help="Build manifest/split stats before training if missing or forced.",
    )
    parser.add_argument("--labels_path", type=Path, default=None, help="Raw labels table path for manifest build.")
    parser.add_argument("--metadata_path", type=Path, default=None, help="Optional metadata table path.")
    parser.add_argument("--column_map_path", type=Path, default=None, help="Optional column map JSON.")
    parser.add_argument("--audio_root", type=Path, default=None, help="Audio root for manifest build.")
    parser.add_argument("--video_root", type=Path, default=None, help="Video root for manifest build.")
    parser.add_argument("--audio_template", type=str, default="{video_id}/{utterance_id}.wav")
    parser.add_argument("--video_template", type=str, default="{video_id}/{utterance_id}.mp4")
    parser.add_argument("--manifest_merge_how", type=str, choices=("left", "inner"), default="left")
    parser.add_argument("--manifest_train_ratio", type=float, default=0.8)
    parser.add_argument("--manifest_val_ratio", type=float, default=0.1)
    parser.add_argument("--manifest_test_ratio", type=float, default=0.1)
    parser.add_argument("--min_duration_sec", type=float, default=1.0)
    parser.add_argument("--max_duration_sec", type=float, default=30.0)
    parser.add_argument("--allow_missing_modalities", action="store_true")
    parser.add_argument(
        "--emotion_vector_has_sentiment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, emotion vector format is [sentiment, happy, sad, angry, surprised, disgust, fearful].",
    )

    # Training hyperparameters.
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--log_every_n_steps", type=int, default=25)
    parser.add_argument("--num_sanity_val_steps", type=int, default=2)

    # Feature extractors / data loading.
    parser.add_argument("--freeze_backbones", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--feature_device",
        type=str,
        default="cpu",
        help="Device for foundation extractors inside Dataset (usually cpu when frozen).",
    )
    parser.add_argument("--audio_model_name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--audio_sampling_rate", type=int, default=16000)
    parser.add_argument("--audio_hidden_state_layer", type=int, default=-1)
    parser.add_argument("--max_audio_seconds", type=float, default=None)
    parser.add_argument("--video_backbone", type=str, choices=("resnet50", "vit_b_16"), default="resnet50")
    parser.add_argument("--num_sampled_frames", type=int, default=16)
    parser.add_argument("--max_num_frames", type=int, default=None)
    parser.add_argument(
        "--read_video_with_seconds",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict_path_check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Collation / sequence control.
    parser.add_argument("--max_audio_tokens", type=int, default=None)
    parser.add_argument("--max_video_tokens", type=int, default=None)
    parser.add_argument("--pad_to_multiple_of", type=int, default=None)

    # Model architecture.
    parser.add_argument("--audio_input_dim", type=int, default=None, help="Optional manual audio input dim.")
    parser.add_argument("--video_input_dim", type=int, default=None, help="Optional manual video input dim.")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2, help="Self-attention refinement layers.")
    parser.add_argument("--ff_multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", type=str, choices=("max", "attn"), default="max")
    parser.add_argument("--fusion", type=str, choices=("concat", "add"), default="concat")

    # LightningModule loss + metrics.
    parser.add_argument("--regression_loss", type=str, choices=("huber", "mse"), default="huber")
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--presence_threshold", type=float, default=0.1)
    parser.add_argument(
        "--class_weight_strategy",
        type=str,
        choices=("inverse_presence", "inverse_sqrt_presence"),
        default="inverse_presence",
    )
    parser.add_argument("--class_weight_min", type=float, default=0.1)
    parser.add_argument("--class_weight_max", type=float, default=20.0)
    parser.add_argument(
        "--normalize_class_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Optimizer/scheduler.
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--optim_eps", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Trainer infra.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--matmul_precision", type=str, choices=("high", "medium", "highest"), default="high")

    # Logging/callbacks.
    parser.add_argument("--log_dir", type=Path, default=Path("logs"))
    parser.add_argument("--experiment_name", type=str, default="mer_avtca")
    parser.add_argument("--experiment_version", type=str, default=None)
    parser.add_argument("--monitor_metric", type=str, choices=("val_mae", "val_pcc"), default="val_mae")
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--save_top_k", type=int, default=1)

    return parser.parse_args()


def resolve_manifest_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    manifest_path = args.manifest_path
    if manifest_path is None:
        manifest_path = args.data_dir / "manifests" / "mosei_manifest.csv"
    manifest_path = Path(manifest_path)

    split_stats_path = args.split_stats_path
    if split_stats_path is None:
        split_stats_path = manifest_path.parent / f"{manifest_path.stem}_split_stats.json"
    split_stats_path = Path(split_stats_path)
    return manifest_path, split_stats_path


def maybe_build_manifest(
    args: argparse.Namespace,
    manifest_path: Path,
    split_stats_path: Path,
) -> None:
    need_build = args.build_manifest or (not manifest_path.exists()) or (not split_stats_path.exists())
    if not need_build:
        LOGGER.info("Using existing manifest and split stats.")
        return

    if args.labels_path is None:
        raise ValueError(
            "Manifest build required but --labels_path is missing. "
            "Provide raw labels path or disable --build_manifest and pass existing manifest/stat files."
        )

    output_format = "csv"
    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        output_format = "jsonl"
    elif suffix not in {".csv", ".jsonl"}:
        raise ValueError("train.py manifest auto-build supports .csv or .jsonl manifest outputs.")

    cfg = PipelineConfig(
        labels_path=Path(args.labels_path),
        metadata_path=Path(args.metadata_path) if args.metadata_path is not None else None,
        output_dir=manifest_path.parent,
        output_stem=manifest_path.stem,
        output_format=output_format,
        column_map_path=Path(args.column_map_path) if args.column_map_path is not None else None,
        audio_root=Path(args.audio_root) if args.audio_root is not None else None,
        video_root=Path(args.video_root) if args.video_root is not None else None,
        audio_template=args.audio_template,
        video_template=args.video_template,
        split_ratios=(args.manifest_train_ratio, args.manifest_val_ratio, args.manifest_test_ratio),
        seed=int(args.seed),
        min_duration_sec=float(args.min_duration_sec),
        max_duration_sec=float(args.max_duration_sec),
        allow_missing_modalities=bool(args.allow_missing_modalities),
        emotion_vector_has_sentiment=bool(args.emotion_vector_has_sentiment),
        merge_how=args.manifest_merge_how,
    )

    LOGGER.info("Building manifest with PipelineConfig...")
    ManifestBuilder(cfg).run()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest build completed but file not found: {manifest_path}")
    if not split_stats_path.exists():
        raise FileNotFoundError(f"Split stats not found after manifest build: {split_stats_path}")


def build_datamodule(args: argparse.Namespace, manifest_path: Path) -> MERDataModule:
    if (not args.freeze_backbones) and args.num_workers > 0:
        LOGGER.warning(
            "freeze_backbones=False with num_workers>0 is unsupported by current dataset pipeline. "
            "Overriding num_workers to 0."
        )
        args.num_workers = 0

    audio_cfg = AudioExtractorConfig(
        model_name=args.audio_model_name,
        sampling_rate=args.audio_sampling_rate,
        device=args.feature_device,
        freeze_backbone=args.freeze_backbones,
        max_audio_seconds=args.max_audio_seconds,
        hidden_state_layer=args.audio_hidden_state_layer,
    )
    video_cfg = VideoExtractorConfig(
        backbone=args.video_backbone,
        device=args.feature_device,
        freeze_backbone=args.freeze_backbones,
        num_sampled_frames=args.num_sampled_frames,
        max_num_frames=args.max_num_frames,
        read_video_with_seconds=args.read_video_with_seconds,
    )

    train_loader_cfg = DataLoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2,
    )
    eval_loader_cfg = DataLoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2,
    )
    collate_cfg = CollateConfig(
        max_audio_tokens=args.max_audio_tokens,
        max_video_tokens=args.max_video_tokens,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )

    datamodule = MERDataModule(
        manifest_path=manifest_path,
        audio_cfg=audio_cfg,
        video_cfg=video_cfg,
        train_loader_cfg=train_loader_cfg,
        eval_loader_cfg=eval_loader_cfg,
        collate_cfg=collate_cfg,
        target_columns=TARGET_COLUMNS,
        strict_path_check=args.strict_path_check,
    )
    return datamodule


def _load_manifest(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def _stable_hash_to_int(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def _compute_split_counts(
    n_groups: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int]:
    if n_groups <= 0:
        return (0, 0, 0)
    if n_groups == 1:
        return (1, 0, 0)
    if n_groups == 2:
        return (1, 1, 0)

    counts = [
        max(1, int(round(n_groups * train_ratio))),
        max(1, int(round(n_groups * val_ratio))),
        max(1, int(round(n_groups * test_ratio))),
    ]
    while sum(counts) > n_groups:
        idx = max(range(3), key=lambda i: counts[i])
        if counts[idx] <= 1:
            break
        counts[idx] -= 1
    while sum(counts) < n_groups:
        idx = max(range(3), key=lambda i: (train_ratio, val_ratio, test_ratio)[i])
        counts[idx] += 1
    return (counts[0], counts[1], counts[2])


def _build_video_split_map(
    video_ids: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, str]:
    unique_ids = sorted({str(v).strip() for v in video_ids if str(v).strip()})
    n = len(unique_ids)
    if n == 0:
        return {}

    sorted_ids = sorted(unique_ids, key=lambda v: _stable_hash_to_int(v, seed))
    n_train, n_val, n_test = _compute_split_counts(n, train_ratio, val_ratio, test_ratio)

    train_ids = sorted_ids[:n_train]
    val_ids = sorted_ids[n_train : n_train + n_val]
    test_ids = sorted_ids[n_train + n_val : n_train + n_val + n_test]

    mapping: Dict[str, str] = {}
    for vid in train_ids:
        mapping[vid] = "train"
    for vid in val_ids:
        mapping[vid] = "val"
    for vid in test_ids:
        mapping[vid] = "test"

    for vid in sorted_ids:
        if vid not in mapping:
            mapping[vid] = "train"
    return mapping


def ensure_precomputed_split_column(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    ratios_sum = train_ratio + val_ratio + test_ratio
    if not torch.isclose(torch.tensor(ratios_sum), torch.tensor(1.0), atol=1e-6):
        raise ValueError(
            "Split ratios must sum to 1.0. "
            f"Got train={train_ratio}, val={val_ratio}, test={test_ratio}."
        )
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("Split ratios must be positive.")

    out = frame.copy()
    if "split" in out.columns:
        split = out["split"].astype("string").str.strip().str.lower()
    else:
        split = pd.Series([pd.NA] * len(out), index=out.index, dtype="string")

    allowed = {"train", "val", "test"}
    needs_fill = split.isna() | (~split.isin(allowed))

    if needs_fill.any():
        mapping = _build_video_split_map(
            video_ids=out["video_id"].astype(str).tolist(),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        inferred = out["video_id"].astype(str).map(mapping)
        split = split.where(~needs_fill, inferred)

    if split.isna().any():
        raise ValueError("Unable to assign split for all rows in precomputed manifest.")
    if (~split.isin(allowed)).any():
        bad_vals = sorted(set(split[~split.isin(allowed)].astype(str).tolist()))
        raise ValueError(f"Invalid split labels found after processing: {bad_vals}")

    out["split"] = split.astype(str)
    return out


def validate_precomputed_manifest_columns(frame: pd.DataFrame) -> None:
    missing = [col for col in PRECOMPUTED_REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(
            "Precomputed-feature mode requires these columns in manifest: "
            f"{list(PRECOMPUTED_REQUIRED_COLUMNS)}. Missing: {missing}"
        )


class PrecomputedFeatureDataset(Dataset[Dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        target_columns: Sequence[str],
        strict_path_check: bool = True,
    ) -> None:
        super().__init__()
        self.df = frame.reset_index(drop=True).copy()
        self.target_columns = list(target_columns)
        self.strict_path_check = strict_path_check
        if len(self.df) == 0:
            raise ValueError("PrecomputedFeatureDataset received an empty dataframe.")

    def __len__(self) -> int:
        return len(self.df)

    def _load_feature_tensor(self, path_value: object, column_name: str, row_idx: int) -> torch.Tensor:
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"Row {row_idx} has invalid {column_name}: {path_value!r}")
        feature_path = Path(path_value)
        if self.strict_path_check and not feature_path.exists():
            raise FileNotFoundError(f"{column_name} does not exist: {feature_path}")

        obj = torch.load(str(feature_path), map_location="cpu")
        if not torch.is_tensor(obj):
            tensor = torch.as_tensor(obj)
        else:
            tensor = obj
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                f"{column_name} must be a 2D [T, D] tensor after load, got shape={tuple(tensor.shape)}"
            )
        return tensor.to(torch.float32)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        row = self.df.iloc[index]
        video_id = str(row["video_id"])
        utterance_id = str(row["utterance_id"])
        speaker_id = ""
        if "speaker_id" in self.df.columns and not pd.isna(row["speaker_id"]):
            speaker_id = str(row["speaker_id"])

        audio_features = self._load_feature_tensor(row["audio_feature_path"], "audio_feature_path", index)
        video_features = self._load_feature_tensor(row["video_feature_path"], "video_feature_path", index)

        labels_np = row[self.target_columns].to_numpy(dtype="float32")
        labels = torch.from_numpy(labels_np)

        return {
            "video_id": video_id,
            "utterance_id": utterance_id,
            "speaker_id": speaker_id,
            "audio_features": audio_features,
            "video_features": video_features,
            "labels": labels,
        }


def build_precomputed_dataloaders(
    args: argparse.Namespace,
    manifest_path: Path,
) -> Tuple[Dict[str, DataLoader[BatchDict]], pd.DataFrame]:
    df = _load_manifest(manifest_path)
    validate_precomputed_manifest_columns(df)

    df = df.copy()
    df["video_id"] = df["video_id"].astype(str)
    df["utterance_id"] = df["utterance_id"].astype(str)
    if "speaker_id" not in df.columns:
        df["speaker_id"] = ""

    df = ensure_precomputed_split_column(
        frame=df,
        train_ratio=float(args.manifest_train_ratio),
        val_ratio=float(args.manifest_val_ratio),
        test_ratio=float(args.manifest_test_ratio),
        seed=int(args.seed),
    )

    split_counts = df["split"].value_counts().to_dict()
    LOGGER.info("Precomputed manifest split counts: %s", split_counts)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty:
        raise ValueError("No train rows found for precomputed-feature mode.")
    if val_df.empty:
        raise ValueError("No val rows found for precomputed-feature mode.")
    if test_df.empty:
        raise ValueError("No test rows found for precomputed-feature mode.")

    train_ds = PrecomputedFeatureDataset(
        frame=train_df,
        target_columns=TARGET_COLUMNS,
        strict_path_check=bool(args.precomputed_strict_path_check),
    )
    val_ds = PrecomputedFeatureDataset(
        frame=val_df,
        target_columns=TARGET_COLUMNS,
        strict_path_check=bool(args.precomputed_strict_path_check),
    )
    test_ds = PrecomputedFeatureDataset(
        frame=test_df,
        target_columns=TARGET_COLUMNS,
        strict_path_check=bool(args.precomputed_strict_path_check),
    )

    collate_cfg = CollateConfig(
        max_audio_tokens=args.max_audio_tokens,
        max_video_tokens=args.max_video_tokens,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )
    collate_fn = MultimodalCollator(collate_cfg)

    def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader[BatchDict]:
        kwargs: Dict[str, object] = {
            "dataset": dataset,
            "batch_size": args.batch_size,
            "shuffle": shuffle,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "drop_last": False,
            "persistent_workers": bool(args.num_workers > 0),
            "collate_fn": collate_fn,
        }
        if args.num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return DataLoader(**kwargs)

    loaders = {
        "train": make_loader(train_ds, shuffle=True),
        "val": make_loader(val_ds, shuffle=False),
        "test": make_loader(test_ds, shuffle=False),
    }
    return loaders, df


def class_weights_from_manifest(
    df: pd.DataFrame,
    target_columns: Sequence[str],
    split: str,
    strategy: str,
    min_weight: float,
    max_weight: float,
    normalize_mean_to_one: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    if "split" not in df.columns:
        raise ValueError("Cannot compute class weights from manifest without a 'split' column.")

    split_df = df[df["split"].astype(str) == split].copy()
    if split_df.empty:
        raise ValueError(f"Cannot compute class weights: split '{split}' has zero rows.")

    ratios = []
    for emotion in target_columns:
        values = split_df[emotion].astype(float).to_numpy()
        ratio = float((values > 0.0).mean()) if len(values) else 0.0
        ratios.append(max(ratio, eps))

    ratio_tensor = torch.tensor(ratios, dtype=torch.float32)
    if strategy == "inverse_presence":
        weights = 1.0 / ratio_tensor
    elif strategy == "inverse_sqrt_presence":
        weights = 1.0 / torch.sqrt(ratio_tensor)
    else:
        raise ValueError(f"Unsupported class weight strategy: {strategy}")

    if normalize_mean_to_one:
        weights = weights / weights.mean().clamp(min=eps)

    weights = weights.clamp(min=min_weight, max=max_weight)
    return weights


def infer_input_dims_from_batch(datamodule: MERDataModule) -> Tuple[int, int]:
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))
    audio_dim = int(batch["audio_features"].shape[-1])
    video_dim = int(batch["video_features"].shape[-1])
    return audio_dim, video_dim


def infer_input_dims_from_loader(loader: DataLoader[BatchDict]) -> Tuple[int, int]:
    batch = next(iter(loader))
    audio_dim = int(batch["audio_features"].shape[-1])
    video_dim = int(batch["video_features"].shape[-1])
    return audio_dim, video_dim


def monitor_mode(metric_name: str) -> str:
    if metric_name == "val_mae":
        return "min"
    if metric_name == "val_pcc":
        return "max"
    raise ValueError(f"Unsupported monitor metric: {metric_name}")


def main() -> None:
    configure_logging()
    args = parse_args()

    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.matmul_precision)

    manifest_path, split_stats_path = resolve_manifest_paths(args)

    datamodule: Optional[MERDataModule] = None
    train_loader: Optional[DataLoader[BatchDict]] = None
    val_loader: Optional[DataLoader[BatchDict]] = None
    test_loader: Optional[DataLoader[BatchDict]] = None

    if args.use_precomputed_features:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Precomputed manifest not found: {manifest_path}")

        loaders, manifest_df = build_precomputed_dataloaders(args=args, manifest_path=manifest_path)
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        test_loader = loaders["test"]

        class_weights = class_weights_from_manifest(
            df=manifest_df,
            target_columns=TARGET_COLUMNS,
            split="train",
            strategy=args.class_weight_strategy,
            min_weight=args.class_weight_min,
            max_weight=args.class_weight_max,
            normalize_mean_to_one=args.normalize_class_weights,
        )
        LOGGER.info("Loaded class weights from manifest: %s", class_weights.tolist())

        audio_dim = args.audio_input_dim
        video_dim = args.video_input_dim
        if audio_dim is None or video_dim is None:
            inferred_audio_dim, inferred_video_dim = infer_input_dims_from_loader(train_loader)
            audio_dim = inferred_audio_dim if audio_dim is None else audio_dim
            video_dim = inferred_video_dim if video_dim is None else video_dim
        LOGGER.info("Using input dims: audio=%d, video=%d", audio_dim, video_dim)
    else:
        maybe_build_manifest(args=args, manifest_path=manifest_path, split_stats_path=split_stats_path)

        datamodule = build_datamodule(args=args, manifest_path=manifest_path)
        datamodule.setup(stage="fit")

        class_weights = MERLightningModule.class_weights_from_split_stats(
            split_stats_path=split_stats_path,
            target_columns=TARGET_COLUMNS,
            split="train",
            strategy=args.class_weight_strategy,
            min_weight=args.class_weight_min,
            max_weight=args.class_weight_max,
            normalize_mean_to_one=args.normalize_class_weights,
        )
        LOGGER.info("Loaded class weights: %s", class_weights.tolist())

        audio_dim = args.audio_input_dim
        video_dim = args.video_input_dim
        if audio_dim is None or video_dim is None:
            inferred_audio_dim, inferred_video_dim = infer_input_dims_from_batch(datamodule)
            audio_dim = inferred_audio_dim if audio_dim is None else audio_dim
            video_dim = inferred_video_dim if video_dim is None else video_dim
        LOGGER.info("Using input dims: audio=%d, video=%d", audio_dim, video_dim)

    model_cfg = AVTCAModelConfig(
        audio_input_dim=audio_dim,
        video_input_dim=video_dim,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        num_self_attn_layers=args.num_layers,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        pooling=args.pooling,
        fusion=args.fusion,
        num_outputs=len(TARGET_COLUMNS),
    )
    model = AVTCAModel(model_cfg)

    optimizer_cfg = OptimizerConfig(
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.optim_eps,
    )
    scheduler_cfg = SchedulerConfig(
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        warmup_start_factor=args.warmup_start_factor,
        eta_min=args.min_lr,
    )

    lightning_module = MERLightningModule(
        model=model,
        class_weights=class_weights,
        optimizer_config=optimizer_cfg,
        scheduler_config=scheduler_cfg,
        regression_loss=args.regression_loss,
        huber_delta=args.huber_delta,
        presence_threshold=args.presence_threshold,
        target_columns=TARGET_COLUMNS,
        sync_dist=True,
    )

    logger = TensorBoardLogger(
        save_dir=str(args.log_dir),
        name=args.experiment_name,
        version=args.experiment_version,
    )
    metric_mode = monitor_mode(args.monitor_metric)

    ckpt_callback = ModelCheckpoint(
        monitor=args.monitor_metric,
        mode=metric_mode,
        save_top_k=args.save_top_k,
        save_last=True,
        filename="{epoch:02d}-{step:06d}",
    )
    early_stopping = EarlyStopping(
        monitor=args.monitor_metric,
        mode=metric_mode,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        deterministic=args.deterministic,
        callbacks=[ckpt_callback, early_stopping, lr_monitor],
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
    )

    if args.use_precomputed_features:
        assert train_loader is not None and val_loader is not None and test_loader is not None
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        trainer.test(model=lightning_module, dataloaders=test_loader, ckpt_path="best")
    else:
        assert datamodule is not None
        trainer.fit(model=lightning_module, datamodule=datamodule)
        trainer.test(model=lightning_module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
