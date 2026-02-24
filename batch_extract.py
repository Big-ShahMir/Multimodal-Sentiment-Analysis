#!/usr/bin/env python3
"""
Resumable micro-batch ETL for CMU-MOSEI under strict local storage constraints.

Design choice:
- Reuse existing project modules where compatible.
  - Download/cropping primitives from `subset_data.py`
  - Foundation extractors from `dataset.py`

Pipeline
1) Read mini_manifest.csv (read-only).
2) Process one video_id at a time:
   - Skip if in failed_downloads.txt
   - Skip if all utterances already in training_manifest.csv
   - Download video once (yt-dlp)
   - Crop each utterance to temporary mp4/wav (ffmpeg)
   - Extract audio/video features via dataset.py extractors
   - Save features as .pt and append row to training_manifest.csv
3) Immediately delete temporary raw and cropped media.
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch

try:
    from .dataset import (
        AudioExtractorConfig,
        AudioFoundationExtractor,
        VideoExtractorConfig,
        VideoFoundationExtractor,
    )
except ImportError:
    try:
        from dataset import (  # type: ignore[no-redef]
            AudioExtractorConfig,
            AudioFoundationExtractor,
            VideoExtractorConfig,
            VideoFoundationExtractor,
        )
    except Exception as exc:  # noqa: BLE001
        AudioExtractorConfig = None  # type: ignore[assignment]
        AudioFoundationExtractor = None  # type: ignore[assignment]
        VideoExtractorConfig = None  # type: ignore[assignment]
        VideoFoundationExtractor = None  # type: ignore[assignment]
        _DATASET_IMPORT_ERROR = exc
    else:
        _DATASET_IMPORT_ERROR = None
else:
    _DATASET_IMPORT_ERROR = None

try:
    from .subset_data import (
        crop_utterance as subset_crop_utterance,
        download_video as subset_download_video,
        ensure_tool_available as subset_ensure_tool_available,
        sanitize_for_filename,
    )
except ImportError:
    try:
        from subset_data import (  # type: ignore[no-redef]
            crop_utterance as subset_crop_utterance,
            download_video as subset_download_video,
            ensure_tool_available as subset_ensure_tool_available,
            sanitize_for_filename,
        )
    except Exception as exc:  # noqa: BLE001
        subset_crop_utterance = None  # type: ignore[assignment]
        subset_download_video = None  # type: ignore[assignment]
        subset_ensure_tool_available = None  # type: ignore[assignment]
        sanitize_for_filename = None  # type: ignore[assignment]
        _SUBSET_IMPORT_ERROR = exc
    else:
        _SUBSET_IMPORT_ERROR = None
else:
    _SUBSET_IMPORT_ERROR = None


REQUIRED_COLUMNS: List[str] = [
    "video_id",
    "utterance_id",
    "start_time",
    "end_time",
    "raw_features",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
    "utterance_dominant_emotion",
]

TRAINING_MANIFEST_NAME = "training_manifest.csv"
FAILED_DOWNLOADS_NAME = "failed_downloads.txt"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable per-video download/crop/feature extraction for a mini MOSEI manifest.",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        required=True,
        help="Path to read-only mini_manifest.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for feature tensors and state files.",
    )
    parser.add_argument(
        "--batch_limit",
        type=int,
        default=0,
        help="Optional max number of videos to process this run (0 = no limit).",
    )
    return parser.parse_args()


def ensure_runtime_dependencies() -> None:
    if _DATASET_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import dataset.py extractor classes. "
            f"Original error: {_DATASET_IMPORT_ERROR}"
        )
    if _SUBSET_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import subset_data.py media utilities. "
            f"Original error: {_SUBSET_IMPORT_ERROR}"
        )
    if (
        AudioExtractorConfig is None
        or AudioFoundationExtractor is None
        or VideoExtractorConfig is None
        or VideoFoundationExtractor is None
    ):
        raise RuntimeError("dataset.py extractor symbols are unavailable.")
    if (
        subset_crop_utterance is None
        or subset_download_video is None
        or subset_ensure_tool_available is None
        or sanitize_for_filename is None
    ):
        raise RuntimeError("subset_data.py utility symbols are unavailable.")


def validate_manifest_columns(frame: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in manifest: {missing}")


def load_failed_video_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    failed: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            video_id = line.strip()
            if video_id:
                failed.add(video_id)
    return failed


def append_failed_video_id(path: Path, video_id: str, failed_cache: Set[str]) -> None:
    if video_id in failed_cache:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{video_id}\n")
    failed_cache.add(video_id)


def load_processed_pairs(path: Path) -> Set[Tuple[str, str]]:
    if not path.exists():
        return set()

    processed: Set[Tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return processed
        if "video_id" not in reader.fieldnames or "utterance_id" not in reader.fieldnames:
            raise ValueError(
                f"{path} exists but does not contain required columns 'video_id' and 'utterance_id'."
            )
        for row in reader:
            video_id = str(row["video_id"]).strip()
            utterance_id = str(row["utterance_id"]).strip()
            if video_id and utterance_id:
                processed.add((video_id, utterance_id))
    return processed


def init_training_manifest(path: Path, input_columns: Sequence[str]) -> List[str]:
    output_columns = list(input_columns) + ["audio_feature_path", "video_feature_path"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_columns)
            writer.writeheader()
    return output_columns


def append_training_row(path: Path, row: Dict[str, object], fieldnames: Sequence[str]) -> None:
    serializable = {col: row.get(col, "") for col in fieldnames}
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writerow(serializable)


def safe_remove(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to remove %s: %s", path, exc)


def get_pending_group_rows(
    group_df: pd.DataFrame,
    processed_pairs: Set[Tuple[str, str]],
) -> pd.DataFrame:
    mask = [
        (str(row_video_id).strip(), str(row_utt_id).strip()) not in processed_pairs
        for row_video_id, row_utt_id in zip(group_df["video_id"], group_df["utterance_id"])
    ]
    return group_df.loc[mask].copy()


def process_single_video(
    video_id: str,
    group_df: pd.DataFrame,
    output_dir: Path,
    training_manifest_path: Path,
    training_columns: Sequence[str],
    processed_pairs: Set[Tuple[str, str]],
    failed_ids: Set[str],
    failed_path: Path,
    audio_extractor: AudioFoundationExtractor,
    video_extractor: VideoFoundationExtractor,
) -> Dict[str, int]:
    stats = {"utterances_ok": 0, "utterances_failed": 0, "download_failed": 0}
    safe_video_id = sanitize_for_filename(video_id)

    tmp_root = output_dir / "_tmp"
    feature_audio_root = output_dir / "features" / "audio" / safe_video_id
    feature_video_root = output_dir / "features" / "video" / safe_video_id
    tmp_video_dir = tmp_root / safe_video_id
    tmp_video_dir.mkdir(parents=True, exist_ok=True)
    feature_audio_root.mkdir(parents=True, exist_ok=True)
    feature_video_root.mkdir(parents=True, exist_ok=True)

    raw_video_path: Optional[Path] = None

    try:
        downloaded_video, download_err = subset_download_video(video_id, tmp_video_dir)
        if downloaded_video is None:
            logging.error("Download failed for %s: %s", video_id, download_err)
            append_failed_video_id(failed_path, video_id, failed_ids)
            stats["download_failed"] = 1
            return stats
        raw_video_path = downloaded_video

        for _, row in group_df.iterrows():
            row_video_id = str(row["video_id"]).strip()
            row_utterance_id = str(row["utterance_id"]).strip()
            row_key = (row_video_id, row_utterance_id)
            if row_key in processed_pairs:
                continue

            start_time = pd.to_numeric(row["start_time"], errors="coerce")
            end_time = pd.to_numeric(row["end_time"], errors="coerce")
            if pd.isna(start_time) or pd.isna(end_time) or float(end_time) <= float(start_time):
                logging.warning(
                    "Skipping invalid timestamps for %s/%s (start=%s, end=%s)",
                    row_video_id,
                    row_utterance_id,
                    row["start_time"],
                    row["end_time"],
                )
                stats["utterances_failed"] += 1
                continue

            safe_utt = sanitize_for_filename(row_utterance_id)
            utt_video_tmp = tmp_video_dir / f"{safe_video_id}__{safe_utt}.mp4"
            utt_audio_tmp = tmp_video_dir / f"{safe_video_id}__{safe_utt}.wav"

            audio_feature_path = feature_audio_root / f"{safe_video_id}__{safe_utt}.pt"
            video_feature_path = feature_video_root / f"{safe_video_id}__{safe_utt}.pt"

            audio_features: Optional[torch.Tensor] = None
            video_features: Optional[torch.Tensor] = None

            try:
                crop_err = subset_crop_utterance(
                    source_video=raw_video_path,
                    start_time=float(start_time),
                    end_time=float(end_time),
                    out_video_path=utt_video_tmp,
                    out_audio_path=utt_audio_tmp,
                )
                if crop_err is not None:
                    raise RuntimeError(crop_err)

                # Clips are already utterance-level; pass no time slicing.
                audio_features = audio_extractor.extract_from_path(
                    audio_path=utt_audio_tmp,
                    start_time=None,
                    end_time=None,
                )
                video_features = video_extractor.extract_from_path(
                    video_path=utt_video_tmp,
                    start_time=None,
                    end_time=None,
                )

                torch.save(audio_features, str(audio_feature_path))
                torch.save(video_features, str(video_feature_path))

                out_row = {col: row[col] for col in training_columns if col in row.index}
                out_row["audio_feature_path"] = str(audio_feature_path.resolve())
                out_row["video_feature_path"] = str(video_feature_path.resolve())
                append_training_row(training_manifest_path, out_row, training_columns)
                processed_pairs.add(row_key)
                stats["utterances_ok"] += 1

            except Exception as exc:  # noqa: BLE001
                logging.error(
                    "Utterance extraction failed for %s/%s: %s",
                    row_video_id,
                    row_utterance_id,
                    exc,
                )
                stats["utterances_failed"] += 1
                safe_remove(audio_feature_path)
                safe_remove(video_feature_path)
            finally:
                safe_remove(utt_video_tmp)
                safe_remove(utt_audio_tmp)

                if video_features is not None:
                    del video_features
                if audio_features is not None:
                    del audio_features
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except Exception as exc:  # noqa: BLE001
        logging.exception("Unhandled error while processing video_id=%s: %s", video_id, exc)
    finally:
        if raw_video_path is not None:
            safe_remove(raw_video_path)
        shutil.rmtree(tmp_video_dir, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return stats


def main() -> None:
    configure_logging()
    args = parse_args()

    ensure_runtime_dependencies()
    subset_ensure_tool_available("yt-dlp")
    subset_ensure_tool_available("ffmpeg")

    manifest_path = args.manifest_path
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    validate_manifest_columns(manifest_df, REQUIRED_COLUMNS)
    manifest_df["video_id"] = manifest_df["video_id"].astype(str)
    manifest_df["utterance_id"] = manifest_df["utterance_id"].astype(str)

    training_manifest_path = output_dir / TRAINING_MANIFEST_NAME
    failed_path = output_dir / FAILED_DOWNLOADS_NAME
    training_columns = init_training_manifest(training_manifest_path, list(manifest_df.columns))

    failed_ids = load_failed_video_ids(failed_path)
    processed_pairs = load_processed_pairs(training_manifest_path)

    feature_device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Feature extraction device: %s", feature_device)

    audio_cfg = AudioExtractorConfig(
        model_name="facebook/wav2vec2-base",
        sampling_rate=16000,
        device=feature_device,
        freeze_backbone=True,
        max_audio_seconds=None,
        hidden_state_layer=-1,
        normalize_waveform=True,
    )
    video_cfg = VideoExtractorConfig(
        backbone="resnet50",
        device=feature_device,
        freeze_backbone=True,
        num_sampled_frames=16,
        max_num_frames=None,
        read_video_with_seconds=False,
    )

    audio_extractor = AudioFoundationExtractor(audio_cfg)
    video_extractor = VideoFoundationExtractor(video_cfg)

    grouped = manifest_df.groupby("video_id", sort=False)
    total_videos = grouped.ngroups
    processed_video_count = 0
    downloaded_video_count = 0
    agg_ok = 0
    agg_utt_failed = 0
    agg_download_failed = 0

    for idx, (video_id, group_df) in enumerate(grouped, start=1):
        video_id = str(video_id).strip()
        if not video_id:
            continue

        if video_id in failed_ids:
            logging.info("[%d/%d] Skipping %s (already in failed log).", idx, total_videos, video_id)
            continue

        pending_df = get_pending_group_rows(group_df, processed_pairs)
        if pending_df.empty:
            logging.info("[%d/%d] Skipping %s (already completed).", idx, total_videos, video_id)
            continue

        if args.batch_limit > 0 and processed_video_count >= args.batch_limit:
            logging.info("batch_limit=%d reached. Stopping.", args.batch_limit)
            break

        if downloaded_video_count > 0:
            sleep_s = random.uniform(10.0, 30.0)
            logging.info("Sleeping %.1f seconds before next download.", sleep_s)
            time.sleep(sleep_s)

        logging.info(
            "[%d/%d] Processing video_id=%s with %d pending utterances",
            idx,
            total_videos,
            video_id,
            len(pending_df),
        )
        stats = process_single_video(
            video_id=video_id,
            group_df=pending_df,
            output_dir=output_dir,
            training_manifest_path=training_manifest_path,
            training_columns=training_columns,
            processed_pairs=processed_pairs,
            failed_ids=failed_ids,
            failed_path=failed_path,
            audio_extractor=audio_extractor,
            video_extractor=video_extractor,
        )

        processed_video_count += 1
        downloaded_video_count += 1
        agg_ok += stats["utterances_ok"]
        agg_utt_failed += stats["utterances_failed"]
        agg_download_failed += stats["download_failed"]

    logging.info("Run complete.")
    logging.info("Videos processed this run: %d", processed_video_count)
    logging.info("Utterances succeeded this run: %d", agg_ok)
    logging.info("Utterances failed this run: %d", agg_utt_failed)
    logging.info("Videos failed download this run: %d", agg_download_failed)
    logging.info("Append-only manifest: %s", training_manifest_path.resolve())
    logging.info("Failed download log: %s", failed_path.resolve())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.error("Interrupted by user.")
        sys.exit(130)
