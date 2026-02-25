#!/usr/bin/env python3
"""
Metadata-first stratified subset builder for CMU-MOSEI manifests.

Workflow
1) Load a full utterance manifest.
2) Derive utterance emotion intensities and dominant emotion.
3) Aggregate to video-level dominant emotion (mean emotion intensity per video).
4) Stratified-sample videos using sklearn.train_test_split(stratify=...).
5) Write mini_manifest.csv with only utterances from sampled videos.
6) Optionally download each sampled YouTube video once with yt-dlp,
   crop per-utterance .mp4 and .wav with ffmpeg, and update local paths.

Example
-------
python subset_data.py ^
  --manifest_path "dataset download/data/full_manifest.csv" ^
  --fraction 0.1 ^
  --output_dir "dataset download/data/subset" ^
  --download
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EMOTION_COLUMNS: List[str] = ["happy", "sad", "angry", "fearful", "disgust", "surprised"]
MINI_MANIFEST_NAME = "mini_manifest.csv"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a stratified MOSEI mini-manifest and optionally download/crop only that subset.",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        required=True,
        help="Path to full manifest CSV.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="Fraction of unique videos to keep (0 < fraction <= 1). Example: 0.1 for 10%%.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for mini_manifest.csv and optional downloaded/cropped files.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="If set, run yt-dlp + ffmpeg to download sampled videos and crop utterance-level media.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def assert_required_columns(frame: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_feature_vector(value: object) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return [float(x) for x in parsed]
    return None


def vector_to_emotions(vector: Optional[List[float]]) -> Dict[str, float]:
    if vector is None:
        return {col: np.nan for col in EMOTION_COLUMNS}

    if len(vector) >= 7:
        # Common MOSEI layout: [sentiment, happy, sad, anger, surprise, disgust, fear]
        mapped = {
            "happy": vector[1],
            "sad": vector[2],
            "angry": vector[3],
            "surprised": vector[4],
            "disgust": vector[5],
            "fearful": vector[6],
        }
        return mapped

    if len(vector) >= 6:
        mapped = {
            "happy": vector[0],
            "sad": vector[1],
            "angry": vector[2],
            "fearful": vector[3],
            "disgust": vector[4],
            "surprised": vector[5],
        }
        return mapped

    return {col: np.nan for col in EMOTION_COLUMNS}


def ensure_emotion_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    # Convert existing explicit emotion columns if present.
    for col in EMOTION_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    missing_emotion_cols = [col for col in EMOTION_COLUMNS if col not in frame.columns]
    needs_fill = bool(missing_emotion_cols)

    if not needs_fill:
        # Even with all columns present, fill remaining NaNs from raw_features if available.
        needs_fill = "raw_features" in frame.columns and frame[EMOTION_COLUMNS].isna().any(axis=None)

    if not needs_fill:
        return frame

    if "raw_features" not in frame.columns:
        raise ValueError(
            "Manifest does not have complete explicit emotion columns and also lacks 'raw_features'."
        )

    vectors = frame["raw_features"].apply(parse_feature_vector)
    mapped = vectors.apply(vector_to_emotions)
    mapped_df = pd.DataFrame(list(mapped), index=frame.index)

    for col in EMOTION_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.to_numeric(mapped_df[col], errors="coerce")
        else:
            frame[col] = frame[col].fillna(pd.to_numeric(mapped_df[col], errors="coerce"))

    return frame


def dominant_label_from_scores(scores: np.ndarray) -> np.ndarray:
    # scores shape: (n_rows, n_emotions)
    if scores.ndim != 2 or scores.shape[1] != len(EMOTION_COLUMNS):
        raise ValueError("scores array has unexpected shape")

    nan_mask = np.isnan(scores)
    all_nan = nan_mask.all(axis=1)

    safe_scores = scores.copy()
    safe_scores[nan_mask] = -np.inf

    best_idx = np.argmax(safe_scores, axis=1)
    labels = np.array([EMOTION_COLUMNS[i] for i in best_idx], dtype=object)
    labels[all_nan] = "unknown"
    return labels


def attach_utterance_dominant_emotion(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    scores = frame[EMOTION_COLUMNS].to_numpy(dtype=float)
    frame["utterance_dominant_emotion"] = dominant_label_from_scores(scores)
    return frame


def build_video_level_table(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("video_id", as_index=False)[EMOTION_COLUMNS].mean()
    grouped_scores = grouped[EMOTION_COLUMNS].to_numpy(dtype=float)
    grouped["video_dominant_emotion"] = dominant_label_from_scores(grouped_scores)
    return grouped


def summarize_distribution(labels: Iterable[object]) -> Dict[str, Dict[str, float]]:
    series = pd.Series(list(labels), dtype="string").fillna("unknown")
    counts = series.value_counts(dropna=False).sort_index()
    total = int(counts.sum())

    out: Dict[str, Dict[str, float]] = {}
    for label, count in counts.items():
        out[str(label)] = {
            "count": int(count),
            "fraction": float(count / total) if total > 0 else 0.0,
        }
    return out


def log_distribution(title: str, labels: Iterable[object]) -> Dict[str, Dict[str, float]]:
    stats = summarize_distribution(labels)
    logging.info(title)
    for label, values in stats.items():
        logging.info("  %-12s count=%6d frac=%.4f", label, values["count"], values["fraction"])
    return stats


def choose_sample_size(total_videos: int, fraction: float) -> int:
    sample_size = int(round(total_videos * fraction))
    sample_size = max(1, sample_size)
    sample_size = min(total_videos, sample_size)
    return sample_size


def sample_video_ids(video_table: pd.DataFrame, fraction: float, seed: int) -> Set[str]:
    total_videos = len(video_table)
    if total_videos == 0:
        return set()

    sample_size = choose_sample_size(total_videos, fraction)
    if sample_size >= total_videos:
        logging.info("Requested fraction selects all videos (%d/%d).", sample_size, total_videos)
        return set(video_table["video_id"].astype(str).tolist())

    labels = video_table["video_dominant_emotion"].astype("string").fillna("unknown")

    # Stratify requires each class to have >=2 members. Collapse rare classes into one bucket.
    class_counts = labels.value_counts()
    rare_classes = set(class_counts[class_counts < 2].index.tolist())
    strat_labels = labels.where(~labels.isin(rare_classes), "__rare__")

    # Stratified split also needs at least one sample per class in both train and test splits.
    n_classes = int(strat_labels.nunique())
    min_test = n_classes
    max_test = total_videos - n_classes

    adjusted_sample_size = sample_size
    if adjusted_sample_size < min_test:
        adjusted_sample_size = min_test
    if max_test >= 1 and adjusted_sample_size > max_test:
        adjusted_sample_size = max_test

    if adjusted_sample_size <= 0 or n_classes <= 1:
        logging.warning(
            "Falling back to random sampling (insufficient class diversity for stratification)."
        )
        return set(
            video_table.sample(n=sample_size, random_state=seed)["video_id"].astype(str).tolist()
        )

    if adjusted_sample_size != sample_size:
        logging.warning(
            "Adjusted sample size from %d to %d to satisfy stratified split constraints.",
            sample_size,
            adjusted_sample_size,
        )

    try:
        _, sampled_ids = train_test_split(
            video_table["video_id"].astype(str),
            test_size=adjusted_sample_size,
            random_state=seed,
            stratify=strat_labels,
        )
        return set(sampled_ids.tolist())
    except ValueError as exc:
        logging.warning("Stratified sampling failed (%s). Falling back to random sampling.", exc)
        return set(
            video_table.sample(n=sample_size, random_state=seed)["video_id"].astype(str).tolist()
        )


def ensure_tool_available(tool_name: str) -> None:
    if shutil.which(tool_name) is None:
        raise RuntimeError(
            f"'{tool_name}' was not found on PATH. Install it and retry with --download."
        )


def sanitize_for_filename(text: object) -> str:
    text_str = str(text).strip()
    if not text_str:
        return "empty"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text_str)
    sanitized = sanitized.strip("._")
    return sanitized or "id"


def run_subprocess(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def infer_downloaded_path(stdout_text: str, download_dir: Path, safe_video_id: str) -> Optional[Path]:
    lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    for line in reversed(lines):
        candidate = Path(line)
        if candidate.exists():
            return candidate

    candidates = sorted(download_dir.glob(f"{safe_video_id}.*"))
    media_candidates = [
        path
        for path in candidates
        if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}
    ]
    if media_candidates:
        return media_candidates[0]
    return None


def download_video(video_id: str, raw_video_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    safe_id = sanitize_for_filename(video_id)

    existing = sorted(raw_video_dir.glob(f"{safe_id}.*"))
    existing_media = [
        path
        for path in existing
        if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}
    ]
    if existing_media:
        return existing_media[0], None

    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(raw_video_dir / f"{safe_id}.%(ext)s")

    command = [
        "yt-dlp",
        "--no-playlist",
        "--no-progress",
        "--newline",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        "--print",
        "after_move:filepath",
        url,
    ]

    result = run_subprocess(command)
    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown yt-dlp error"
        return None, error_msg

    downloaded_path = infer_downloaded_path(result.stdout, raw_video_dir, safe_id)
    if downloaded_path is None:
        return None, "yt-dlp completed, but output file could not be located"

    return downloaded_path, None


def crop_utterance(
    source_video: Path,
    start_time: float,
    end_time: float,
    out_video_path: Path,
    out_audio_path: Path,
) -> Optional[str]:
    duration = float(end_time) - float(start_time)
    if duration <= 0:
        return "Invalid interval: end_time must be greater than start_time"

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    out_audio_path.parent.mkdir(parents=True, exist_ok=True)

    # Re-encode for precise utterance boundaries.
    video_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_video),
        "-ss",
        f"{start_time:.3f}",
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        str(out_video_path),
    ]
    video_result = run_subprocess(video_cmd)
    if video_result.returncode != 0:
        return video_result.stderr.strip() or "ffmpeg video crop failed"

    audio_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_video),
        "-ss",
        f"{start_time:.3f}",
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_audio_path),
    ]
    audio_result = run_subprocess(audio_cmd)
    if audio_result.returncode != 0:
        return audio_result.stderr.strip() or "ffmpeg audio crop failed"

    return None


def download_and_crop_subset(mini_manifest: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    ensure_tool_available("yt-dlp")
    ensure_tool_available("ffmpeg")

    raw_video_dir = output_dir / "raw_videos"
    clipped_video_dir = output_dir / "utterances" / "video"
    clipped_audio_dir = output_dir / "utterances" / "audio"

    raw_video_dir.mkdir(parents=True, exist_ok=True)
    clipped_video_dir.mkdir(parents=True, exist_ok=True)
    clipped_audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = mini_manifest.copy()
    if "video_path" not in manifest.columns:
        manifest["video_path"] = ""
    if "audio_path" not in manifest.columns:
        manifest["audio_path"] = ""
    manifest["download_status"] = "pending"
    manifest["download_error"] = ""

    grouped = manifest.groupby("video_id", sort=False)
    total_videos = grouped.ngroups

    for idx_video, (video_id, group) in enumerate(grouped, start=1):
        logging.info("[%d/%d] Processing video_id=%s", idx_video, total_videos, video_id)

        downloaded_video, download_error = download_video(str(video_id), raw_video_dir)
        if downloaded_video is None:
            msg = f"download_failed: {download_error}"
            logging.warning("  %s", msg)
            manifest.loc[group.index, "download_status"] = "download_failed"
            manifest.loc[group.index, "download_error"] = str(download_error)
            continue

        for row_index in group.index:
            start_time = pd.to_numeric(manifest.at[row_index, "start_time"], errors="coerce")
            end_time = pd.to_numeric(manifest.at[row_index, "end_time"], errors="coerce")

            if pd.isna(start_time) or pd.isna(end_time):
                manifest.at[row_index, "download_status"] = "invalid_timestamps"
                manifest.at[row_index, "download_error"] = "start_time/end_time is NaN"
                continue

            if float(end_time) <= float(start_time):
                manifest.at[row_index, "download_status"] = "invalid_timestamps"
                manifest.at[row_index, "download_error"] = "end_time <= start_time"
                continue

            utterance_id = manifest.at[row_index, "utterance_id"] if "utterance_id" in manifest.columns else row_index
            base_name = (
                f"{sanitize_for_filename(video_id)}__"
                f"{sanitize_for_filename(utterance_id)}__"
                f"{int(row_index)}"
            )
            out_video = clipped_video_dir / f"{base_name}.mp4"
            out_audio = clipped_audio_dir / f"{base_name}.wav"

            if out_video.exists() and out_audio.exists():
                manifest.at[row_index, "video_path"] = str(out_video.resolve())
                manifest.at[row_index, "audio_path"] = str(out_audio.resolve())
                manifest.at[row_index, "download_status"] = "ok"
                manifest.at[row_index, "download_error"] = ""
                continue

            crop_error = crop_utterance(
                source_video=downloaded_video,
                start_time=float(start_time),
                end_time=float(end_time),
                out_video_path=out_video,
                out_audio_path=out_audio,
            )

            if crop_error is None:
                manifest.at[row_index, "video_path"] = str(out_video.resolve())
                manifest.at[row_index, "audio_path"] = str(out_audio.resolve())
                manifest.at[row_index, "download_status"] = "ok"
                manifest.at[row_index, "download_error"] = ""
            else:
                manifest.at[row_index, "download_status"] = "crop_failed"
                manifest.at[row_index, "download_error"] = crop_error

    return manifest


def run_pipeline(args: argparse.Namespace) -> None:
    if not (0 < args.fraction <= 1.0):
        raise ValueError("--fraction must be in (0, 1].")

    if not args.manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {args.manifest_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    mini_manifest_path = output_dir / MINI_MANIFEST_NAME
    stats_path = output_dir / "subset_stats.json"

    manifest = pd.read_csv(args.manifest_path)
    assert_required_columns(manifest, ["video_id", "start_time", "end_time"])
    if "utterance_id" not in manifest.columns:
        manifest["utterance_id"] = manifest.index.astype(str)

    logging.info("Loaded manifest with %d utterances.", len(manifest))

    manifest = ensure_emotion_columns(manifest)
    manifest = attach_utterance_dominant_emotion(manifest)
    video_table = build_video_level_table(manifest)

    full_utt_stats = log_distribution(
        "Full manifest utterance dominant emotion distribution:",
        manifest["utterance_dominant_emotion"],
    )
    full_video_stats = log_distribution(
        "Full manifest video dominant emotion distribution:",
        video_table["video_dominant_emotion"],
    )

    sampled_video_ids = sample_video_ids(video_table, fraction=args.fraction, seed=args.seed)
    mini_manifest = manifest[manifest["video_id"].astype(str).isin(sampled_video_ids)].copy()

    mini_video_table = video_table[video_table["video_id"].astype(str).isin(sampled_video_ids)].copy()

    subset_utt_stats = log_distribution(
        "Subset utterance dominant emotion distribution:",
        mini_manifest["utterance_dominant_emotion"],
    )
    subset_video_stats = log_distribution(
        "Subset video dominant emotion distribution:",
        mini_video_table["video_dominant_emotion"],
    )

    stats_payload = {
        "full": {
            "num_utterances": int(len(manifest)),
            "num_videos": int(len(video_table)),
            "utterance_distribution": full_utt_stats,
            "video_distribution": full_video_stats,
        },
        "subset": {
            "num_utterances": int(len(mini_manifest)),
            "num_videos": int(len(mini_video_table)),
            "fraction_requested": float(args.fraction),
            "fraction_realized": float(len(mini_video_table) / len(video_table)) if len(video_table) else 0.0,
            "utterance_distribution": subset_utt_stats,
            "video_distribution": subset_video_stats,
        },
    }

    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2)
    logging.info("Saved subset stats to %s", stats_path)

    mini_manifest.to_csv(mini_manifest_path, index=False)
    logging.info("Saved mini manifest to %s", mini_manifest_path)

    if args.download:
        logging.info("--download enabled. Starting selective yt-dlp/ffmpeg pipeline...")
        mini_manifest = download_and_crop_subset(mini_manifest, output_dir=output_dir)
        mini_manifest.to_csv(mini_manifest_path, index=False)
        logging.info("Updated mini manifest with local media paths at %s", mini_manifest_path)

        status_counts = mini_manifest["download_status"].value_counts(dropna=False).to_dict()
        logging.info("Download/crop status summary: %s", status_counts)


def main() -> int:
    configure_logging()
    try:
        args = parse_args()
        run_pipeline(args)
        return 0
    except Exception as exc:
        logging.exception("subset_data.py failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

