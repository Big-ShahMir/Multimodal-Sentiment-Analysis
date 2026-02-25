#!/usr/bin/env python3
"""
cleanup_failed_features.py

Deletes empty audio/video feature directories that were created for
videos that failed to download during batch_extract.py runs.

Safety checks performed before any deletion:
  - Directory must be empty (no .pt files inside at any depth).
  - video_id must appear in failed_downloads.txt.
  - The corresponding entry must NOT appear in training_manifest.csv
    (i.e. no utterances were ever successfully extracted for that video).

The ETL pipeline (batch_extract.py) is safe to re-run after cleanup:
it recreates feature dirs with mkdir(exist_ok=True) before each attempt.

Usage:
    python3 cleanup_failed_features.py --features_dir <path/to/features> [--dry-run]

    --dry-run   Print what would be deleted without actually deleting anything.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_failed_ids(failed_path: Path) -> set[str]:
    if not failed_path.exists():
        logging.warning("failed_downloads.txt not found: %s", failed_path)
        return set()
    ids: set[str] = set()
    for line in failed_path.read_text(encoding="utf-8").splitlines():
        vid = line.strip()
        if vid:
            ids.add(vid)
    return ids


def load_successful_ids(manifest_path: Path) -> set[str]:
    """Return video_ids that have at least one successfully extracted utterance."""
    if not manifest_path.exists():
        return set()
    ids: set[str] = set()
    with manifest_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            vid = str(row.get("video_id", "")).strip()
            if vid:
                ids.add(vid)
    return ids


def is_empty_feature_dir(dir_path: Path) -> bool:
    """Return True if the directory contains no .pt files at any depth."""
    return not any(dir_path.rglob("*.pt"))


def cleanup(features_dir: Path, dry_run: bool) -> None:
    failed_path = features_dir / "failed_downloads.txt"
    manifest_path = features_dir / "training_manifest.csv"

    failed_ids = load_failed_ids(failed_path)
    successful_ids = load_successful_ids(manifest_path)

    if not failed_ids:
        logging.info("No failed video IDs found. Nothing to clean up.")
        return

    logging.info("Failed video IDs: %s", sorted(failed_ids))
    logging.info("Successfully extracted video IDs: %s", sorted(successful_ids))

    audio_root = features_dir / "features" / "audio"
    video_root = features_dir / "features" / "video"

    deleted_count = 0
    skipped_count = 0

    for video_id in sorted(failed_ids):
        audio_dir = audio_root / video_id
        video_dir = video_root / video_id

        # Safety: skip if this video also has successful utterances.
        if video_id in successful_ids:
            logging.warning(
                "Skipping %s — it appears in both failed_downloads.txt AND "
                "training_manifest.csv. Manual review recommended.",
                video_id,
            )
            skipped_count += 1
            continue

        for feature_dir in (audio_dir, video_dir):
            if not feature_dir.exists():
                logging.debug("Already absent, skipping: %s", feature_dir)
                continue

            if not is_empty_feature_dir(feature_dir):
                logging.warning(
                    "Skipping %s — it contains .pt files despite being in "
                    "failed_downloads.txt. Manual review recommended.",
                    feature_dir,
                )
                skipped_count += 1
                continue

            if dry_run:
                logging.info("[DRY RUN] Would delete: %s", feature_dir)
            else:
                shutil.rmtree(feature_dir)
                logging.info("Deleted: %s", feature_dir)
            deleted_count += 1

    if dry_run:
        logging.info(
            "Dry run complete. Would delete %d director%s, skipped %d.",
            deleted_count,
            "y" if deleted_count == 1 else "ies",
            skipped_count,
        )
    else:
        logging.info(
            "Cleanup complete. Deleted %d director%s, skipped %d.",
            deleted_count,
            "y" if deleted_count == 1 else "ies",
            skipped_count,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete empty feature dirs left by failed batch_extract.py downloads.",
    )
    parser.add_argument(
        "--features_dir",
        type=Path,
        required=True,
        help="Path to the features output directory (contains failed_downloads.txt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be deleted without actually deleting anything.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()

    if not args.features_dir.exists():
        logging.error("features_dir does not exist: %s", args.features_dir)
        sys.exit(1)

    cleanup(features_dir=args.features_dir, dry_run=args.dry_run)
