# Multimodal Emotion Recognition (MER) with Bidirectional Cross-Attention

## Overview
This repository contains a state-of-the-art (SOTA) Multimodal Emotion Recognition (MER) pipeline trained on the CMU-MOSEI dataset. The CMU-MOSEI dataset contains more than 23,500 annotated YouTube monologue segments. 

To effectively process this data, our model introduces an intermediate transformer-based fusion mechanism. By abandoning legacy static features (like COVAREP and FACET) in favor of dynamic foundation models, and replacing naive data truncation with imbalance-aware loss functions, this architecture robustly predicts continuous emotion intensities across six core categories: Happy, Sad, Angry, Fearful, Disgust, and Surprised.

## Key Features & Upgrades
* **Full Data Utilization:** We keep all valid utterances instead of deleting data to downsample majority classes. Extreme dataset imbalance is handled algorithmically via Inverse-Sqrt Weighted Huber and MSE loss functions.
* **Dynamic Foundation Extractors:** Audio is processed dynamically using Hugging Face `Wav2Vec2` or `HuBERT` backbones. Video is processed using `ViT` or `ResNet50` backbones with uniform frame sampling.
* **Bidirectional Cross-Attention Fusion:** Processing both interactions in parallel creates bidirectional mapping. Audio queries attend to Video key/values, and Video queries attend to Audio key/values simultaneously, followed by a temporal self-attention refinement layer.
* **SOTA Evaluation Metrics:** We evaluate using Mean Absolute Error (MAE), Pearson Correlation Coefficient (PCC), and Top-K Accuracy to natively handle naturally co-occurring emotions. This avoids the metric degradation seen when coarse binary thresholding amplifies the effects of class imbalance.

## Repository Structure
The codebase is organized into functional subpackages:

* **`etl/`**: Data collection and feature extraction pipeline.
    * `build_manifest.py`: Cleans raw CMU-MOSEI data and enforces splits.
    * `batch_extract.py`: Main ETL script for downloading and extracting features.
    * `subset_data.py`: Utility for creating smaller data subsets and managing downloads.
    * `cleanup_failed_features.py`: Safety utility to remove empty artifacts from failed downloads.
* **`data/`**: Core PyTorch data handling.
    * `dataset.py`: `MultimodalManifestDataset` and dynamic foundation extractors.
    * `data_loader.py`: DataLoader factories and custom sequence collation.
    * `datamodule.py`: PyTorch Lightning interface for split orchestration.
* **`model/`**: Architecture and training logic.
    * `model.py`: `AVTCAModel` cross-attention fusion architecture.
    * `lightning_module.py`: PL wrapper with imbalance-aware losses and metrics.
* **`train.py`**: Main CLI entry point for model training.

## Installation & Setup
1. Clone this repository and navigate to the root directory.
2. Install Python dependencies: `pip install -r requirements.txt`
3. **External Dependencies**:
    * **FFmpeg**: Required for video/audio processing. 
    * **yt-dlp**: Required for video downloading.
    Ensure `ffmpeg` and `ffprobe` are in your `PATH`.

## ETL Pipeline (Feature Extraction)
To process the CMU-MOSEI dataset and precompute features:

1. **Prepare Manifest**: Generate a subset manifest if you don't want to process the full 23k+ utterances.
   ```bash
   python etl/subset_data.py --fraction 0.02 --output_dir ./data/subset_2pct
   ```

2. **Run Batch Extraction**: Download videos and extract audio/video features to `.pt` files.
   ```bash
   python etl/batch_extract.py \
     --manifest_path "./data/subset_2pct/mini_manifest.csv" \
     --output_dir "./data/subset_2pct/features" \
     --batch_limit 100
   ```

3. **Cleanup (Optional)**: If any downloads failed, clean up the empty directory markers.
   ```bash
   python etl/cleanup_failed_features.py --features_dir "./data/subset_2pct/features"
   ```

## Usage (Training)
### On-the-fly Extraction
To train while extracting features in real-time (requires high disk I/O):
```bash
python train.py --data_dir ./data --batch_size 4 --max_epochs 50
```

### From Precomputed Features
To train using features generated in the ETL step (recommended for speed/reproducibility):
```bash
python train.py \
  --use_precomputed_features \
  --manifest_path "./data/subset_2pct/features/training_manifest.csv" \
  --batch_size 16
```