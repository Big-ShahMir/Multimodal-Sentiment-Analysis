#!/usr/bin/env python3
"""
Dataset and feature extraction primitives for multimodal emotion recognition.

This module provides:
1. Foundation-model audio feature extraction (Wav2Vec2 / HuBERT / compatible HF models).
2. Foundation-model visual feature extraction (ResNet50 / ViT from torchvision).
3. A manifest-driven PyTorch Dataset that dynamically loads audio/video per sample.

The expected manifest schema includes:
    video_id, utterance_id, speaker_id, start_time, end_time, audio_path, video_path,
    happy, sad, angry, fearful, disgust, surprised, split
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, TypedDict

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

try:
    import torchaudio
except Exception:  # pragma: no cover - optional dependency at runtime
    torchaudio = None  # type: ignore[assignment]

try:
    import soundfile as _soundfile
except Exception:  # pragma: no cover - optional dependency at runtime
    _soundfile = None  # type: ignore[assignment]


def _load_audio_sf(path: Path) -> tuple:
    """Load audio via soundfile (libsndfile), returning (waveform [C,T] float32 Tensor, sample_rate int).

    Falls back to torchaudio.load() if soundfile is unavailable.
    Avoids torchaudio 2.9+ which routes load() through torchcodec.
    """
    if _soundfile is not None:
        import numpy as np
        data, sr = _soundfile.read(str(path), dtype="float32", always_2d=True)  # [T, C]
        waveform = torch.from_numpy(np.ascontiguousarray(data.T))  # [C, T]
        return waveform, sr
    if torchaudio is not None:
        return torchaudio.load(path)
    raise ImportError("Neither soundfile nor torchaudio is available for audio loading.")

try:
    import torchvision
    from torchvision.io import read_video
    from torchvision.models import (
        ResNet50_Weights,
        ViT_B_16_Weights,
        resnet50,
        vit_b_16,
    )
    # Force the PyAV backend for read_video.  The default C++ backend requires
    # system-level FFmpeg shared libraries (libavutil.so) which are not available
    # in this environment; PyAV (the 'av' pip package) ships its own FFmpeg and
    # works without system libs.
    torchvision.set_video_backend("pyav")
except Exception:  # pragma: no cover - optional dependency at runtime
    read_video = None  # type: ignore[assignment]
    ResNet50_Weights = None  # type: ignore[assignment]
    ViT_B_16_Weights = None  # type: ignore[assignment]
    resnet50 = None  # type: ignore[assignment]
    vit_b_16 = None  # type: ignore[assignment]

try:
    from transformers import AutoFeatureExtractor, AutoModel
except Exception:  # pragma: no cover - optional dependency at runtime
    AutoFeatureExtractor = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

TARGET_COLUMNS: List[str] = ["happy", "sad", "angry", "fearful", "disgust", "surprised"]
REQUIRED_MANIFEST_COLUMNS: List[str] = [
    "video_id",
    "utterance_id",
    "speaker_id",
    "start_time",
    "end_time",
    "audio_path",
    "video_path",
    "split",
    *TARGET_COLUMNS,
]


class DatasetSample(TypedDict):
    video_id: str
    utterance_id: str
    speaker_id: str
    audio_features: Tensor
    video_features: Tensor
    labels: Tensor


@dataclass(frozen=True)
class AudioExtractorConfig:
    model_name: str = "facebook/wav2vec2-base"
    sampling_rate: int = 16_000
    device: str = "cpu"
    freeze_backbone: bool = True
    max_audio_seconds: Optional[float] = None
    hidden_state_layer: int = -1
    normalize_waveform: bool = True


@dataclass(frozen=True)
class VideoExtractorConfig:
    backbone: Literal["resnet50", "vit_b_16"] = "resnet50"
    device: str = "cpu"
    freeze_backbone: bool = True
    num_sampled_frames: int = 16
    max_num_frames: Optional[int] = None
    read_video_with_seconds: bool = True


def _load_manifest(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        try:
            df = pd.read_json(path)
        except ValueError:
            df = pd.read_json(path, lines=True)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}")
    return df


def _validate_manifest_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")


class AudioFoundationExtractor(nn.Module):
    """
    Dynamic audio feature extraction with a HF foundation model.

    Typical outputs:
        Wav2Vec2 base -> [T_audio_tokens, 768]
        HuBERT base   -> [T_audio_tokens, 768]
    """

    def __init__(self, config: AudioExtractorConfig) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        if AutoFeatureExtractor is None or AutoModel is None:
            raise ImportError(
                "transformers is required for AudioFoundationExtractor. "
                "Install with: pip install transformers"
            )
        if torchaudio is None:
            raise ImportError(
                "torchaudio is required for AudioFoundationExtractor. "
                "Install with: pip install torchaudio"
            )

        self.processor = AutoFeatureExtractor.from_pretrained(config.model_name)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.to(self.device)

        if config.freeze_backbone:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def freeze_backbone(self) -> bool:
        return self.config.freeze_backbone

    def _load_waveform(self, audio_path: Path) -> Tensor:
        waveform, sample_rate = _load_audio_sf(audio_path)
        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform [channels, time], got shape={tuple(waveform.shape)}")

        # Convert to mono for robust ASR-style encoders.
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.config.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.config.sampling_rate,
            )

        if self.config.normalize_waveform:
            # Keep normalization simple and stable to avoid clipping.
            denom = waveform.abs().max().clamp(min=1e-8)
            waveform = waveform / denom

        return waveform

    def _slice_waveform(self, waveform: Tensor, start_time: float, end_time: float) -> Tensor:
        if start_time < 0:
            start_time = 0.0
        if end_time <= start_time:
            return waveform
        sr = self.config.sampling_rate
        start_idx = int(round(start_time * sr))
        end_idx = int(round(end_time * sr))
        end_idx = min(end_idx, waveform.shape[-1])
        start_idx = max(0, min(start_idx, end_idx))
        sliced = waveform[:, start_idx:end_idx]
        if sliced.numel() == 0:
            return waveform
        return sliced

    def extract_from_path(
        self,
        audio_path: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Tensor:
        """
        Extract contextualized acoustic features for one utterance.

        Returns:
            Tensor with shape [T_audio_tokens, D_audio]
        """
        waveform = self._load_waveform(audio_path)

        if start_time is not None and end_time is not None:
            waveform = self._slice_waveform(waveform, start_time=start_time, end_time=end_time)

        if self.config.max_audio_seconds is not None:
            max_samples = int(round(self.config.max_audio_seconds * self.config.sampling_rate))
            waveform = waveform[:, :max_samples]

        # HF feature extractors expect a 1D waveform per sample.
        wav_1d = waveform.squeeze(0).contiguous()
        model_inputs = self.processor(
            wav_1d.cpu().numpy(),
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
            padding=False,
        )
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        with torch.set_grad_enabled(not self.config.freeze_backbone):
            if self.config.freeze_backbone:
                with torch.no_grad():
                    outputs = self.model(**model_inputs, output_hidden_states=True)
            else:
                outputs = self.model(**model_inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        if hidden_states is not None:
            features = hidden_states[self.config.hidden_state_layer]  # [1, T, D]
        else:
            features = outputs.last_hidden_state  # [1, T, D]

        features = features.squeeze(0)
        if self.config.freeze_backbone:
            features = features.detach().cpu()
        return features


class VideoFoundationExtractor(nn.Module):
    """
    Dynamic visual feature extraction from sampled utterance frames.

    - `resnet50`: returns [T_video, 2048]
    - `vit_b_16`: returns [T_video, 768]
    """

    def __init__(self, config: VideoExtractorConfig) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        if read_video is None or resnet50 is None or vit_b_16 is None:
            raise ImportError(
                "torchvision with video + model APIs is required for VideoFoundationExtractor. "
                "Install with: pip install torchvision"
            )

        if config.backbone == "resnet50":
            if ResNet50_Weights is None:
                raise ImportError("ResNet50_Weights unavailable in installed torchvision version.")
            weights = ResNet50_Weights.DEFAULT
            base = resnet50(weights=weights)
            # Remove classification head; keep global pooled embedding.
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self.preprocess = weights.transforms()
        elif config.backbone == "vit_b_16":
            if ViT_B_16_Weights is None:
                raise ImportError("ViT_B_16_Weights unavailable in installed torchvision version.")
            weights = ViT_B_16_Weights.DEFAULT
            base = vit_b_16(weights=weights)
            base.heads = nn.Identity()
            self.backbone = base
            self.preprocess = weights.transforms()
        else:  # pragma: no cover - protected by Literal typing
            raise ValueError(f"Unsupported backbone: {config.backbone}")

        self.backbone.to(self.device)
        if config.freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    @property
    def freeze_backbone(self) -> bool:
        return self.config.freeze_backbone

    @staticmethod
    def _ensure_tchw(video_tensor: Tensor) -> Tensor:
        # read_video typically returns [T, H, W, C] for THWC.
        if video_tensor.ndim != 4:
            raise ValueError(f"Expected 4D video tensor, got shape={tuple(video_tensor.shape)}")
        if video_tensor.shape[-1] in (1, 3):
            return video_tensor.permute(0, 3, 1, 2).contiguous()
        return video_tensor.contiguous()

    def _load_video(
        self,
        video_path: Path,
        start_time: Optional[float],
        end_time: Optional[float],
    ) -> Tensor:
        try:
            if self.config.read_video_with_seconds and start_time is not None and end_time is not None:
                video, _, _ = read_video(
                    str(video_path),
                    start_pts=float(start_time),
                    end_pts=float(end_time),
                    pts_unit="sec",
                    output_format="THWC",
                )
                # If timestamps are absolute but file is already utterance-clipped, retry full read.
                if video.numel() == 0:
                    video, _, _ = read_video(str(video_path), output_format="THWC")
            else:
                video, _, _ = read_video(str(video_path), output_format="THWC")
        except Exception as exc:
            raise RuntimeError(f"Failed to read video file: {video_path}") from exc

        if video.numel() == 0:
            raise ValueError(f"No video frames found after decode: {video_path}")
        return self._ensure_tchw(video)

    def _sample_frames(self, frames_tchw: Tensor) -> Tensor:
        t = frames_tchw.shape[0]
        if self.config.max_num_frames is not None:
            t = min(t, self.config.max_num_frames)
            frames_tchw = frames_tchw[:t]
        n = min(max(1, self.config.num_sampled_frames), t)

        # Uniformly sample while allowing duplicates for short clips.
        indices = np.linspace(0, t - 1, num=n, dtype=np.int64)
        sampled = frames_tchw[torch.from_numpy(indices)]
        return sampled

    def extract_from_path(
        self,
        video_path: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Tensor:
        """
        Extract frame-wise visual embeddings for one utterance.

        Returns:
            Tensor with shape [T_video_tokens, D_video]
        """
        frames = self._load_video(video_path, start_time=start_time, end_time=end_time)
        frames = self._sample_frames(frames)

        # Normalize to [0, 1] float before torchvision weights transforms.
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        else:
            frames = frames.float()

        processed = torch.stack([self.preprocess(frame) for frame in frames], dim=0).to(self.device)

        with torch.set_grad_enabled(not self.config.freeze_backbone):
            if self.config.freeze_backbone:
                with torch.no_grad():
                    features = self.backbone(processed)
            else:
                features = self.backbone(processed)

        if self.config.backbone == "resnet50":
            # [T, 2048, 1, 1] -> [T, 2048]
            features = features.flatten(start_dim=1)

        if self.config.freeze_backbone:
            features = features.detach().cpu()
        return features


class MultimodalManifestDataset(Dataset[DatasetSample]):
    """
    Manifest-driven multimodal dataset for CMU-MOSEI-style training.

    Notes
    -----
    - Dynamic loading: files are read in `__getitem__`, not preloaded.
    - This class expects utterance-level paths in the manifest.
    - For split leakage prevention, the split assignment should come from `build_manifest.py`.
    """

    def __init__(
        self,
        manifest_path: Path,
        audio_extractor: AudioFoundationExtractor,
        video_extractor: VideoFoundationExtractor,
        split: Optional[Literal["train", "val", "test"]] = None,
        target_columns: Sequence[str] = TARGET_COLUMNS,
        strict_path_check: bool = True,
    ) -> None:
        super().__init__()
        self.manifest_path = manifest_path
        self.audio_extractor = audio_extractor
        self.video_extractor = video_extractor
        self.target_columns = list(target_columns)
        self.strict_path_check = strict_path_check

        df = _load_manifest(manifest_path)
        _validate_manifest_columns(df)

        if split is not None:
            df = df[df["split"] == split].copy()

        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows available for split={split!r} in {manifest_path}")

        self.df = df
        LOGGER.info(
            "Initialized dataset from %s with %d rows (split=%s)",
            manifest_path,
            len(self.df),
            split,
        )

    def __len__(self) -> int:
        return len(self.df)

    def _get_path(self, value: object, column_name: str, row_idx: int) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Row {row_idx} has invalid {column_name}: {value!r}")
        path = Path(value)
        if self.strict_path_check and not path.exists():
            raise FileNotFoundError(f"{column_name} does not exist: {path}")
        return path

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
            return float(value)
        except Exception:
            return None

    def __getitem__(self, index: int) -> DatasetSample:
        row = self.df.iloc[index]
        video_id = str(row["video_id"])
        utterance_id = str(row["utterance_id"])
        speaker_id = "" if pd.isna(row["speaker_id"]) else str(row["speaker_id"])

        audio_path = self._get_path(row["audio_path"], "audio_path", index)
        video_path = self._get_path(row["video_path"], "video_path", index)

        start_time = self._safe_float(row["start_time"])
        end_time = self._safe_float(row["end_time"])

        try:
            audio_features = self.audio_extractor.extract_from_path(
                audio_path=audio_path,
                start_time=start_time,
                end_time=end_time,
            )
            video_features = self.video_extractor.extract_from_path(
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Feature extraction failed for video_id={video_id}, "
                f"utterance_id={utterance_id} at dataset index {index}."
            ) from exc

        labels_np = row[self.target_columns].to_numpy(dtype=np.float32)
        labels = torch.from_numpy(labels_np)

        return DatasetSample(
            video_id=video_id,
            utterance_id=utterance_id,
            speaker_id=speaker_id,
            audio_features=audio_features,
            video_features=video_features,
            labels=labels,
        )
