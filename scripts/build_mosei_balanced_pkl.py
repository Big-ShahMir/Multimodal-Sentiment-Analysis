#!/usr/bin/env python3
"""
Build a balanced CMU-MOSEI subset and save as one .pkl.

- Loads labels, acoustic (COVAREP), and visual (Facet42) .csd files.
- Maps each segment to one of 6 emotions (dominant = argmax).
- Samples the SAME number of segments per emotion (balanced).
- Splits into train/val/test (80/10/10) and saves one .pkl.

Usage (from repo root):
  pip install mmsdk  # or: pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
  python scripts/build_mosei_balanced_pkl.py --csd_dir data/cmu_mosei_csd --output_path data/mosei_balanced.pkl --samples_per_emotion 500

Required .csd files in --csd_dir:
  - CMU_MOSEI_Labels.csd
  - CMU_MOSEI_COVAREP.csd
  - CMU_MOSEI_VisualFacet42.csd
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)

# Match repo's target order
EMOTION_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprised"]
# Common MOSEI label key variants (different sources use different names)
EMOTION_ALIASES = {
    "happy": ["happy", "happiness"],
    "sad": ["sad", "sadness"],
    "angry": ["angry", "anger"],
    "fearful": ["fearful", "fear"],
    "disgust": ["disgust"],
    "surprised": ["surprised", "surprise"],
}


def _get_mmsdk():
    try:
        from mmsdk import mmdatasdk
        return mmdatasdk
    except ImportError:
        raise ImportError(
            "CMU Multimodal SDK is required. Install with:\n"
            "  pip install mmsdk\n"
            "  or: pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK"
        )


def load_csd_files(csd_dir: Path) -> Tuple[Any, Any, Any]:
    """Load labels, acoustic, visual .csd into mmdatasdk mmdataset objects."""
    mmdatasdk = _get_mmsdk()
    csd_dir = Path(csd_dir)
    labels_path = csd_dir / "CMU_MOSEI_Labels.csd"
    acoustic_path = csd_dir / "CMU_MOSEI_COVAREP.csd"
    visual_path = csd_dir / "CMU_MOSEI_VisualFacet42.csd"
    for p in (labels_path, acoustic_path, visual_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Download the .csd files (see scripts/README_CSD_DOWNLOAD.md).")
    dataset = mmdatasdk.mmdataset(
        {
            "labels": str(labels_path),
            "acoustic": str(acoustic_path),
            "visual": str(visual_path),
        }
    )
    # SDK returns one mmdataset; access computational sequences by name
    if hasattr(dataset, "computational_sequences"):
        cs = dataset.computational_sequences
        return cs.get("labels", dataset), cs.get("acoustic", dataset), cs.get("visual", dataset)
    return dataset["labels"], dataset["acoustic"], dataset["visual"]


def extract_emotion_vector_from_label(
    label_entry: Any, num_emotions: int = 6
) -> Optional[np.ndarray]:
    """
    Get a 6-dim emotion vector from one segment's label entry.
    label_entry is typically from the SDK: may have 'features' array or named keys.
    MOSEI labels often: [sentiment, happy, sad, anger, fear, disgust, surprise] or similar.
    """
    if label_entry is None:
        return None
    # SDK entry may be object with .features
    if hasattr(label_entry, "features"):
        label_entry = {"features": label_entry.features}
    elif not isinstance(label_entry, dict):
        label_entry = {"features": np.asarray(label_entry)}
    # Try 'features' key (array)
    if "features" in label_entry:
        arr = np.array(label_entry["features"])
        if arr.ndim == 2:
            arr = arr.squeeze()
        if arr.size >= num_emotions:
            # Often first column is sentiment; next 6 are emotions
            if arr.size == 7:
                return np.asarray(arr[1:7], dtype=np.float32)
            return np.asarray(arr[:num_emotions], dtype=np.float32)
        return None
    # Try explicit keys
    vec = []
    for emo in EMOTION_NAMES:
        for key in EMOTION_ALIASES.get(emo, [emo]):
            if key in label_entry:
                val = label_entry[key]
                if hasattr(val, "__len__") and len(val) > 0:
                    val = val[0] if not isinstance(val, (int, float)) else val
                vec.append(float(val))
                break
        else:
            vec.append(0.0)
    if len(vec) == num_emotions:
        return np.array(vec, dtype=np.float32)
    return None


def get_segment_emotion_vectors(labels_data: Any) -> List[Tuple[str, np.ndarray]]:
    """
    Iterate labels and return (segment_id, emotion_vector) for each segment.
    labels_data is the SDK computational sequence for labels.
    """
    out: List[Tuple[str, np.ndarray]] = []
    # SDK often exposes: .computational_sequences["labels"].data keyed by segment id
    if hasattr(labels_data, "computational_sequences"):
        cs = labels_data.computational_sequences.get("labels", labels_data)
    else:
        cs = labels_data
    if hasattr(cs, "data"):
        data = cs.data
    else:
        data = cs
    for seg_id, entry in data.items():
        if hasattr(entry, "keys"):
            entry = dict(entry) if not isinstance(entry, dict) else entry
        elif hasattr(entry, "__array__"):
            entry = {"features": np.asarray(entry)}
        else:
            entry = {"features": entry}
        vec = extract_emotion_vector_from_label(entry)
        if vec is not None:
            out.append((seg_id, vec))
    return out


def dominant_emotion_index(vec: np.ndarray) -> int:
    """Argmax over 6 emotions; tie-break by first max."""
    return int(np.argmax(vec))


def balanced_sample(
    segment_vectors: List[Tuple[str, np.ndarray]],
    samples_per_emotion: int,
    seed: int = 42,
) -> List[str]:
    """
    Choose segments so each of the 6 emotions has exactly samples_per_emotion (or fewer if not enough).
    Returns list of segment IDs.
    """
    random.seed(seed)
    np.random.seed(seed)
    by_emotion: Dict[int, List[str]] = {i: [] for i in range(6)}
    for seg_id, vec in segment_vectors:
        idx = dominant_emotion_index(vec)
        by_emotion[idx].append(seg_id)
    # Cap at minimum count so we don't require more than we have
    counts = [len(by_emotion[i]) for i in range(6)]
    n = min(samples_per_emotion, min(counts))
    if n == 0:
        raise ValueError("At least one emotion has no segments. Check labels .csd.")
    chosen: List[str] = []
    for i in range(6):
        chosen.extend(random.sample(by_emotion[i], n))
    random.shuffle(chosen)
    return chosen


def get_features_for_segment(
    acoustic_data: Any,
    visual_data: Any,
    segment_id: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract acoustic and visual feature arrays for one segment from SDK data."""
    def _get_cs_data(cs):
        if hasattr(cs, "data"):
            return cs.data
        return cs
    a_data = _get_cs_data(acoustic_data)
    v_data = _get_cs_data(visual_data)
    audio = None
    video = None
    if segment_id in a_data:
        entry = a_data[segment_id]
        if hasattr(entry, "features"):
            audio = np.asarray(entry.features, dtype=np.float32)
        elif isinstance(entry, dict) and "features" in entry:
            audio = np.asarray(entry["features"], dtype=np.float32)
    if segment_id in v_data:
        entry = v_data[segment_id]
        if hasattr(entry, "features"):
            video = np.asarray(entry.features, dtype=np.float32)
        elif isinstance(entry, dict) and "features" in entry:
            video = np.asarray(entry["features"], dtype=np.float32)
    return audio, video


def build_splits(
    segment_ids: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split segment IDs into train / val / test."""
    random.seed(seed)
    ids = segment_ids.copy()
    random.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def run(
    csd_dir: Path,
    output_path: Path,
    samples_per_emotion: int = 500,
    seed: int = 42,
) -> None:
    csd_dir = Path(csd_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading .csd files from %s ...", csd_dir)
    labels_cs, acoustic_cs, visual_cs = load_csd_files(csd_dir)

    # Wrap if we got the full dataset dict
    if hasattr(labels_cs, "computational_sequences"):
        labels_cs = labels_cs.computational_sequences.get("labels", labels_cs)
    if hasattr(acoustic_cs, "computational_sequences"):
        acoustic_cs = acoustic_cs.computational_sequences.get("acoustic", acoustic_cs)
    if hasattr(visual_cs, "computational_sequences"):
        visual_cs = visual_cs.computational_sequences.get("visual", visual_cs)

    LOG.info("Extracting segment emotion vectors from labels ...")
    segment_vectors = get_segment_emotion_vectors(labels_cs)
    LOG.info("Found %d segments with valid emotion labels.", len(segment_vectors))

    segment_ids = balanced_sample(segment_vectors, samples_per_emotion, seed=seed)
    LOG.info("Balanced subset: %d segments (%d per emotion).", len(segment_ids), len(segment_ids) // 6)

    seg_to_vec = {seg_id: vec for seg_id, vec in segment_vectors}
    splits = build_splits(segment_ids, seed=seed)

    result = {"train": [], "val": [], "test": [], "emotion_names": EMOTION_NAMES}
    for split_name, ids in splits.items():
        LOG.info("Building %s split (%d segments) ...", split_name, len(ids))
        for seg_id in ids:
            audio, video = get_features_for_segment(acoustic_cs, visual_cs, seg_id)
            if audio is None or video is None:
                continue
            labels = seg_to_vec[seg_id]
            result[split_name].append({
                "segment_id": seg_id,
                "audio_features": audio,
                "video_features": video,
                "labels": labels,
            })
        LOG.info("%s: %d samples.", split_name, len(result[split_name]))

    import pickle
    with open(output_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    LOG.info("Saved balanced subset to %s", output_path)


def main():
    p = argparse.ArgumentParser(description="Build balanced CMU-MOSEI subset .pkl from .csd files.")
    p.add_argument("--csd_dir", type=Path, required=True, help="Directory containing the 3 .csd files.")
    p.add_argument("--output_path", type=Path, default=Path("data/mosei_balanced.pkl"), help="Output .pkl path.")
    p.add_argument("--samples_per_emotion", type=int, default=500, help="Number of segments to keep per emotion (balanced).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(
        csd_dir=args.csd_dir,
        output_path=args.output_path,
        samples_per_emotion=args.samples_per_emotion,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
