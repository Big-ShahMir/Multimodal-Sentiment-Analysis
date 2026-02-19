#!/usr/bin/env python3
"""
Build a cleaned, leakage-safe CMU-MOSEI manifest for multimodal training.

This script:
1. Loads a labels table (and optional metadata table).
2. Normalizes schema to canonical MER fields.
3. Cleans invalid records without class-balancing truncation.
4. Enforces group-independent split assignment (speaker-first, video fallback).
5. Writes reproducible manifest files (CSV / JSONL).
6. Logs and exports split-wise label distribution statistics.

Example
-------
python build_manifest.py ^
    --labels-path data/raw/mosei_labels.csv ^
    --metadata-path data/raw/mosei_metadata.csv ^
    --audio-root data/raw/audio ^
    --video-root data/raw/video ^
    --output-dir data/manifests ^
    --seed 42
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EMOTION_COLUMNS: List[str] = ["happy", "sad", "angry", "fearful", "disgust", "surprised"]

# Common aliases seen in MOSEI exports and legacy scripts.
COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "video_id": ("video_id", "video", "clip_id", "segment_id", "id"),
    "utterance_id": ("utterance_id", "utterance", "segment_index", "utt_id", "segment"),
    "speaker_id": ("speaker_id", "speaker", "person_id", "spk_id"),
    "start_time": ("start_time", "start", "start_sec", "start_timestamp"),
    "end_time": ("end_time", "end", "end_sec", "end_timestamp"),
    "audio_path": ("audio_path", "wav_path", "audio_file", "audio"),
    "video_path": ("video_path", "mp4_path", "video_file", "video"),
    "emotion_vector": ("emotion_vector", "labels", "features", "label_vector"),
    "happy": ("happy", "happiness"),
    "sad": ("sad", "sadness"),
    "angry": ("angry", "anger"),
    "fearful": ("fearful", "fear"),
    "disgust": ("disgust",),
    "surprised": ("surprised", "surprise"),
}


@dataclass(frozen=True)
class PipelineConfig:
    labels_path: Path
    metadata_path: Optional[Path]
    output_dir: Path
    output_stem: str
    output_format: str
    column_map_path: Optional[Path]
    audio_root: Optional[Path]
    video_root: Optional[Path]
    audio_template: str
    video_template: str
    split_ratios: Tuple[float, float, float]
    seed: int
    min_duration_sec: float
    max_duration_sec: float
    allow_missing_modalities: bool
    emotion_vector_has_sentiment: bool
    merge_how: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format for {path}. Use csv/tsv/jsonl/parquet.")


def load_column_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Column map must be a JSON object of canonical_name -> source_name.")
    return {str(k): str(v) for k, v in payload.items()}


def _normalize_column_lookup(columns: Iterable[str]) -> Dict[str, str]:
    return {str(col).strip().lower(): str(col) for col in columns}


def resolve_column(
    df: pd.DataFrame,
    canonical_name: str,
    explicit_map: Dict[str, str],
    required: bool = False,
) -> Optional[str]:
    col_lookup = _normalize_column_lookup(df.columns)
    if canonical_name in explicit_map:
        requested = explicit_map[canonical_name].strip().lower()
        if requested in col_lookup:
            return col_lookup[requested]
        raise ValueError(
            f"Column map points '{canonical_name}' to '{explicit_map[canonical_name]}', "
            f"but that column was not found."
        )

    for alias in COLUMN_ALIASES.get(canonical_name, (canonical_name,)):
        hit = col_lookup.get(alias.strip().lower())
        if hit is not None:
            return hit

    if required:
        raise ValueError(
            f"Required canonical field '{canonical_name}' was not found in columns: "
            f"{list(df.columns)}"
        )
    return None


def parse_emotion_vector(value: object) -> Optional[List[float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
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


def stable_hash_to_int(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest, 16)


class ManifestBuilder:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.column_map = load_column_map(config.column_map_path)

    def run(self) -> None:
        labels = load_table(self.config.labels_path)
        logging.info("Loaded labels table: %d rows x %d cols", labels.shape[0], labels.shape[1])

        if self.config.metadata_path is not None:
            metadata = load_table(self.config.metadata_path)
            logging.info(
                "Loaded metadata table: %d rows x %d cols",
                metadata.shape[0],
                metadata.shape[1],
            )
        else:
            metadata = None

        manifest = self._build_canonical_table(labels, metadata)
        manifest = self._clean_records(manifest)
        manifest = self._assign_split(manifest)
        self._write_outputs(manifest)
        stats = compute_split_statistics(manifest, EMOTION_COLUMNS)
        stats_path = self.config.output_dir / f"{self.config.output_stem}_split_stats.json"
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        logging.info("Saved split statistics to %s", stats_path)

    def _build_canonical_table(
        self,
        labels_df: pd.DataFrame,
        metadata_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        df = labels_df.copy()

        video_col = resolve_column(df, "video_id", self.column_map, required=True)
        utt_col = resolve_column(df, "utterance_id", self.column_map, required=False)
        spk_col = resolve_column(df, "speaker_id", self.column_map, required=False)
        start_col = resolve_column(df, "start_time", self.column_map, required=True)
        end_col = resolve_column(df, "end_time", self.column_map, required=True)
        audio_col = resolve_column(df, "audio_path", self.column_map, required=False)
        video_path_col = resolve_column(df, "video_path", self.column_map, required=False)
        vector_col = resolve_column(df, "emotion_vector", self.column_map, required=False)

        out = pd.DataFrame(index=df.index)
        out["video_id"] = df[video_col].astype(str).str.strip()
        out["start_time"] = pd.to_numeric(df[start_col], errors="coerce")
        out["end_time"] = pd.to_numeric(df[end_col], errors="coerce")

        if utt_col is not None:
            out["utterance_id"] = df[utt_col].astype(str).str.strip()
        else:
            out["utterance_id"] = np.nan

        if spk_col is not None:
            out["speaker_id"] = df[spk_col].astype(str).str.strip()
        else:
            out["speaker_id"] = np.nan

        if audio_col is not None:
            out["audio_path"] = df[audio_col].astype(str).str.strip()
        else:
            out["audio_path"] = np.nan

        if video_path_col is not None:
            out["video_path"] = df[video_path_col].astype(str).str.strip()
        else:
            out["video_path"] = np.nan

        for emo in EMOTION_COLUMNS:
            emo_col = resolve_column(df, emo, self.column_map, required=False)
            if emo_col is not None:
                out[emo] = pd.to_numeric(df[emo_col], errors="coerce")
            else:
                out[emo] = np.nan

        if vector_col is not None:
            vectors = df[vector_col].apply(parse_emotion_vector)
            for idx in vectors.index:
                vec = vectors.at[idx]
                if vec is None:
                    continue
                if self.config.emotion_vector_has_sentiment:
                    # MOSEI often stores [sentiment, happy, sad, anger, surprise, disgust, fear].
                    if len(vec) < 7:
                        continue
                    mapped = {
                        "happy": vec[1],
                        "sad": vec[2],
                        "angry": vec[3],
                        "surprised": vec[4],
                        "disgust": vec[5],
                        "fearful": vec[6],
                    }
                else:
                    if len(vec) < 6:
                        continue
                    mapped = {
                        "happy": vec[0],
                        "sad": vec[1],
                        "angry": vec[2],
                        "fearful": vec[3],
                        "disgust": vec[4],
                        "surprised": vec[5],
                    }
                for emo, value in mapped.items():
                    if pd.isna(out.at[idx, emo]):
                        out.at[idx, emo] = float(value)

        if metadata_df is not None:
            out = self._merge_metadata(out, metadata_df)

        out = self._ensure_utterance_id(out)
        out = self._resolve_modality_paths(out)

        return out

    def _merge_metadata(self, out: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
        meta = metadata_df.copy()
        video_col = resolve_column(meta, "video_id", self.column_map, required=True)
        utt_col = resolve_column(meta, "utterance_id", self.column_map, required=False)
        spk_col = resolve_column(meta, "speaker_id", self.column_map, required=False)
        audio_col = resolve_column(meta, "audio_path", self.column_map, required=False)
        video_path_col = resolve_column(meta, "video_path", self.column_map, required=False)

        merge_cols = {"video_id": meta[video_col].astype("string").str.strip()}
        if utt_col is not None:
            merge_cols["utterance_id"] = meta[utt_col].astype("string").str.strip()
        if spk_col is not None:
            merge_cols["speaker_id_meta"] = meta[spk_col].astype("string").str.strip()
        if audio_col is not None:
            merge_cols["audio_path_meta"] = meta[audio_col].astype("string").str.strip()
        if video_path_col is not None:
            merge_cols["video_path_meta"] = meta[video_path_col].astype("string").str.strip()

        meta_subset = pd.DataFrame(merge_cols)
        keys = ["video_id"] + (["utterance_id"] if "utterance_id" in meta_subset.columns else [])
        merged = out.merge(meta_subset.drop_duplicates(keys), on=keys, how=self.config.merge_how)

        if "speaker_id_meta" in merged.columns:
            merged["speaker_id"] = merged["speaker_id"].fillna(merged["speaker_id_meta"])
            merged = merged.drop(columns=["speaker_id_meta"])
        if "audio_path_meta" in merged.columns:
            merged["audio_path"] = merged["audio_path"].fillna(merged["audio_path_meta"])
            merged = merged.drop(columns=["audio_path_meta"])
        if "video_path_meta" in merged.columns:
            merged["video_path"] = merged["video_path"].fillna(merged["video_path_meta"])
            merged = merged.drop(columns=["video_path_meta"])

        return merged

    def _ensure_utterance_id(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        utt_series = frame["utterance_id"].astype("string")
        missing = utt_series.isna() | (utt_series.str.len().fillna(0) == 0)
        if missing.any():
            # Reproducible fallback ID: sorted within each video by start/end/index.
            temp = frame.loc[missing, ["video_id", "start_time", "end_time"]].copy()
            temp["orig_index"] = temp.index
            temp = temp.sort_values(["video_id", "start_time", "end_time", "orig_index"])
            temp["utt_idx"] = temp.groupby("video_id").cumcount()
            generated = (
                temp["video_id"].astype(str)
                + "_utt_"
                + temp["utt_idx"].astype(int).astype(str).str.zfill(5)
            )
            frame.loc[temp["orig_index"], "utterance_id"] = generated.values
            logging.info("Generated fallback utterance_id for %d rows.", missing.sum())
        return frame

    def _resolve_modality_paths(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()

        def resolve_path(raw_value: object, root: Optional[Path], template: str, row: pd.Series) -> Optional[str]:
            if isinstance(raw_value, str) and raw_value.strip():
                candidate = Path(raw_value.strip())
                if not candidate.is_absolute() and root is not None:
                    candidate = root / candidate
                return str(candidate)
            if root is None:
                return None
            rel = template.format(video_id=row["video_id"], utterance_id=row["utterance_id"])
            return str(root / rel)

        frame["audio_path"] = frame.apply(
            lambda r: resolve_path(r["audio_path"], self.config.audio_root, self.config.audio_template, r),
            axis=1,
        )
        frame["video_path"] = frame.apply(
            lambda r: resolve_path(r["video_path"], self.config.video_root, self.config.video_template, r),
            axis=1,
        )
        return frame

    def _clean_records(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        initial_count = len(frame)

        frame["video_id"] = frame["video_id"].astype("string").str.strip()
        frame["utterance_id"] = frame["utterance_id"].astype("string").str.strip()
        frame["speaker_id"] = frame["speaker_id"].astype("string").str.strip()
        frame["speaker_id"] = frame["speaker_id"].replace({"": pd.NA})

        # Drop obvious corruption / malformed metadata rows.
        keep_mask = np.ones(len(frame), dtype=bool)
        keep_mask &= frame["video_id"].notna().to_numpy()
        keep_mask &= frame["utterance_id"].notna().to_numpy()
        keep_mask &= frame["start_time"].notna().to_numpy()
        keep_mask &= frame["end_time"].notna().to_numpy()
        keep_mask &= (frame["end_time"] > frame["start_time"]).to_numpy()

        # Label sanitation.
        label_block = frame[EMOTION_COLUMNS]
        all_labels_nan = label_block.isna().all(axis=1)
        keep_mask &= ~all_labels_nan.to_numpy()

        dropped = int((~keep_mask).sum())
        frame = frame.loc[keep_mask].copy()
        if dropped:
            logging.info("Dropped %d corrupted/invalid rows.", dropped)

        # Keep partial labels, but mark and fill NaN with 0.0 (absence prior).
        frame["had_partial_nan_label"] = frame[EMOTION_COLUMNS].isna().any(axis=1)
        frame[EMOTION_COLUMNS] = frame[EMOTION_COLUMNS].fillna(0.0)

        # Clamp out-of-bound labels to valid [0, 3].
        lower, upper = 0.0, 3.0
        arr = frame[EMOTION_COLUMNS].to_numpy(dtype=np.float32)
        out_of_range = (arr < lower) | (arr > upper)
        frame["had_out_of_range_label"] = out_of_range.any(axis=1)
        arr = np.clip(arr, lower, upper)
        frame[EMOTION_COLUMNS] = arr

        # Duration handling: keep all valid rows, but add standardization fields.
        frame["duration_raw_sec"] = frame["end_time"] - frame["start_time"]
        frame["is_short_utterance"] = frame["duration_raw_sec"] < self.config.min_duration_sec
        frame["is_long_utterance"] = frame["duration_raw_sec"] > self.config.max_duration_sec
        frame["duration_sec"] = frame["duration_raw_sec"].clip(
            lower=self.config.min_duration_sec,
            upper=self.config.max_duration_sec,
        )
        frame["duration_was_clipped"] = (
            frame["duration_sec"].round(8) != frame["duration_raw_sec"].round(8)
        )
        frame["end_time_clipped"] = frame["start_time"] + frame["duration_sec"]

        # Modality path validation.
        frame["audio_exists"] = frame["audio_path"].apply(self._path_exists)
        frame["video_exists"] = frame["video_path"].apply(self._path_exists)
        if self.config.allow_missing_modalities:
            missing_modality = (~frame["audio_exists"]) | (~frame["video_exists"])
            if missing_modality.any():
                logging.warning(
                    "Keeping %d rows with missing modality files due to --allow-missing-modalities.",
                    int(missing_modality.sum()),
                )
        else:
            valid_files = frame["audio_exists"] & frame["video_exists"]
            removed_missing_files = int((~valid_files).sum())
            frame = frame.loc[valid_files].copy()
            if removed_missing_files:
                logging.info(
                    "Dropped %d rows due to missing audio/video files.",
                    removed_missing_files,
                )

        logging.info(
            "Cleaned manifest size: %d -> %d rows (retained %.2f%%).",
            initial_count,
            len(frame),
            100.0 * len(frame) / max(initial_count, 1),
        )
        return frame

    @staticmethod
    def _path_exists(path_value: object) -> bool:
        if not isinstance(path_value, str) or not path_value.strip():
            return False
        return Path(path_value).exists()

    def _assign_split(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()

        split_names = ("train", "val", "test")
        train_r, val_r, test_r = self.config.split_ratios
        if not np.isclose(train_r + val_r + test_r, 1.0):
            raise ValueError("Split ratios must sum to 1.0.")

        group_source = frame["speaker_id"].where(frame["speaker_id"].notna(), frame["video_id"])
        frame["split_group_id"] = group_source.astype(str)

        group_sizes = (
            frame.groupby("split_group_id", sort=False)
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
            .reset_index(drop=True)
        )
        group_sizes["hash"] = group_sizes["split_group_id"].apply(
            lambda x: stable_hash_to_int(x, self.config.seed)
        )
        group_sizes = group_sizes.sort_values(["n", "hash"], ascending=[False, True]).reset_index(drop=True)

        total = len(frame)
        targets = {
            "train": total * train_r,
            "val": total * val_r,
            "test": total * test_r,
        }
        current = {k: 0 for k in split_names}
        assignment: Dict[str, str] = {}

        for row in group_sizes.itertuples(index=False):
            group = row.split_group_id
            size = int(row.n)
            deficits = {
                split: (targets[split] - current[split]) / max(targets[split], 1.0)
                for split in split_names
            }
            # Prioritize highest normalized deficit; if all saturated, choose least over-target.
            if max(deficits.values()) > 0:
                chosen = max(deficits.items(), key=lambda kv: kv[1])[0]
            else:
                chosen = min(
                    split_names,
                    key=lambda s: (current[s] - targets[s], current[s] / max(targets[s], 1.0)),
                )
            assignment[group] = chosen
            current[chosen] += size

        frame["split"] = frame["split_group_id"].map(assignment)
        self._assert_no_group_leakage(frame)

        logging.info(
            "Split counts: train=%d val=%d test=%d",
            int((frame["split"] == "train").sum()),
            int((frame["split"] == "val").sum()),
            int((frame["split"] == "test").sum()),
        )
        return frame

    @staticmethod
    def _assert_no_group_leakage(df: pd.DataFrame) -> None:
        split_groups = {
            split: set(df.loc[df["split"] == split, "split_group_id"].astype(str).unique())
            for split in ("train", "val", "test")
        }
        overlap_tv = split_groups["train"].intersection(split_groups["val"])
        overlap_tt = split_groups["train"].intersection(split_groups["test"])
        overlap_vt = split_groups["val"].intersection(split_groups["test"])
        if overlap_tv or overlap_tt or overlap_vt:
            raise RuntimeError(
                "Group leakage detected between splits. "
                f"train-val={len(overlap_tv)}, train-test={len(overlap_tt)}, val-test={len(overlap_vt)}"
            )

    def _write_outputs(self, manifest: pd.DataFrame) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        out = manifest.copy()

        required_cols = [
            "video_id",
            "utterance_id",
            "speaker_id",
            "start_time",
            "end_time",
            "audio_path",
            "video_path",
            *EMOTION_COLUMNS,
            "split",
        ]
        # Keep required columns first, then include helpful debug/cleaning fields.
        remaining = [c for c in out.columns if c not in required_cols]
        out = out[required_cols + remaining]

        if self.config.output_format in {"csv", "both"}:
            csv_path = self.config.output_dir / f"{self.config.output_stem}.csv"
            out.to_csv(csv_path, index=False)
            logging.info("Saved CSV manifest to %s", csv_path)

        if self.config.output_format in {"jsonl", "both"}:
            jsonl_path = self.config.output_dir / f"{self.config.output_stem}.jsonl"
            out.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
            logging.info("Saved JSONL manifest to %s", jsonl_path)


def compute_split_statistics(df: pd.DataFrame, emotion_cols: Sequence[str]) -> Dict[str, object]:
    """
    Compute distributional stats used later for class-balanced/focal loss weighting.
    """
    result: Dict[str, object] = {"overall_rows": int(len(df)), "splits": {}}
    bins = np.array([0.0, 1.0, 2.0, 3.000001], dtype=np.float64)
    bin_names = ["[0,1)", "[1,2)", "[2,3]"]

    for split_name, split_df in df.groupby("split", dropna=False):
        split_name = str(split_name)
        split_stats: Dict[str, object] = {"rows": int(len(split_df)), "emotions": {}}
        for emo in emotion_cols:
            values = split_df[emo].astype(float).to_numpy()
            hist = np.histogram(values, bins=bins)[0].astype(int)
            split_stats["emotions"][emo] = {
                "presence_count_gt0": int((values > 0.0).sum()),
                "presence_ratio_gt0": float((values > 0.0).mean()) if len(values) else 0.0,
                "mean": float(np.mean(values)) if len(values) else 0.0,
                "std": float(np.std(values)) if len(values) else 0.0,
                "min": float(np.min(values)) if len(values) else 0.0,
                "max": float(np.max(values)) if len(values) else 0.0,
                "p25": float(np.quantile(values, 0.25)) if len(values) else 0.0,
                "p50": float(np.quantile(values, 0.50)) if len(values) else 0.0,
                "p75": float(np.quantile(values, 0.75)) if len(values) else 0.0,
                "intensity_histogram": {name: int(count) for name, count in zip(bin_names, hist)},
            }
        result["splits"][split_name] = split_stats

    return result


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Build a cleaned, speaker-independent CMU-MOSEI manifest.",
    )
    parser.add_argument("--labels-path", type=Path, required=True, help="Path to raw labels table.")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Optional metadata table path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Output directory for manifest artifacts.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="mosei_manifest",
        help="Filename stem for output files.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="both",
        choices=("csv", "jsonl", "both"),
        help="Manifest output format.",
    )
    parser.add_argument(
        "--column-map-path",
        type=Path,
        default=None,
        help=(
            "Optional JSON path for canonical->source column mapping. "
            "Canonical keys include: video_id, utterance_id, speaker_id, start_time, "
            "end_time, audio_path, video_path, emotion_vector, happy, sad, angry, "
            "fearful, disgust, surprised."
        ),
    )
    parser.add_argument("--audio-root", type=Path, default=None, help="Root dir to resolve/build audio paths.")
    parser.add_argument("--video-root", type=Path, default=None, help="Root dir to resolve/build video paths.")
    parser.add_argument(
        "--audio-template",
        type=str,
        default="{video_id}/{utterance_id}.wav",
        help="Template used when audio_path is missing.",
    )
    parser.add_argument(
        "--video-template",
        type=str,
        default="{video_id}/{utterance_id}.mp4",
        help="Template used when video_path is missing.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split assignment.")
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=1.0,
        help="Short-utterance threshold used for flags and clipped duration.",
    )
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=30.0,
        help="Long-utterance threshold used for flags and clipped duration.",
    )
    parser.add_argument(
        "--allow-missing-modalities",
        action="store_true",
        help="Keep rows with missing audio/video paths or files (not recommended for training).",
    )
    parser.add_argument(
        "--emotion-vector-has-sentiment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether emotion_vector starts with sentiment then 6 emotions.",
    )
    parser.add_argument(
        "--merge-how",
        type=str,
        default="left",
        choices=("left", "inner"),
        help="Join mode for merging labels with metadata.",
    )

    args = parser.parse_args()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(r <= 0 for r in ratios):
        raise ValueError("All split ratios must be positive.")

    return PipelineConfig(
        labels_path=args.labels_path,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        output_format=args.output_format,
        column_map_path=args.column_map_path,
        audio_root=args.audio_root,
        video_root=args.video_root,
        audio_template=args.audio_template,
        video_template=args.video_template,
        split_ratios=ratios,
        seed=args.seed,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        allow_missing_modalities=args.allow_missing_modalities,
        emotion_vector_has_sentiment=args.emotion_vector_has_sentiment,
        merge_how=args.merge_how,
    )


def main() -> None:
    configure_logging()
    cfg = parse_args()
    logging.info("Starting manifest build with seed=%d", cfg.seed)
    builder = ManifestBuilder(cfg)
    builder.run()
    logging.info("Manifest build complete.")


if __name__ == "__main__":
    main()
