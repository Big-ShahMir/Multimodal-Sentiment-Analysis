# Multimodal Emotion Recognition (MER) with Bidirectional Cross-Attention

## Overview
This repository contains a state-of-the-art (SOTA) Multimodal Emotion Recognition (MER) pipeline trained on the CMU-MOSEI dataset. 

The CMU-MOSEI dataset contains more than 23,500 annotated YouTube monologue segments. To effectively process this data, our model introduces an intermediate transformer-based fusion mechanism. [cite_start]By abandoning legacy static features (like COVAREP and FACET) in favor of dynamic foundation models, and replacing naive data truncation with imbalance-aware loss functions, this architecture robustly predicts continuous emotion intensities across six core categories: Happy, Sad, Angry, Fearful, Disgust, and Surprised.

## Key Features & Upgrades
* **Full Data Utilization:** We keep all valid utterances instead of deleting data to downsample majority classes. Extreme dataset imbalance is handled algorithmically via Inverse-Sqrt Weighted Huber and MSE loss functions.
* **Dynamic Foundation Extractors:** Audio is processed dynamically using Hugging Face `Wav2Vec2` or `HuBERT` backbones. Video is processed using `ViT` or `ResNet50` backbones with uniform frame sampling.
* **Bidirectional Cross-Attention Fusion:** Processing both interactions in parallel creates bidirectional mapping. Audio queries attend to Video key/values, and Video queries attend to Audio key/values simultaneously, followed by a temporal self-attention refinement layer.
* **SOTA Evaluation Metrics:** We evaluate using Mean Absolute Error (MAE), Pearson Correlation Coefficient (PCC), and Top-K Accuracy to natively handle naturally co-occurring emotions. This avoids the metric degradation seen when coarse binary thresholding amplifies the effects of class imbalance.

## Repository Structure
* **`build_manifest.py`**: The initial ingestion script that cleans raw CMU-MOSEI data, enforces speaker-independent splits, and calculates class frequency statistics for the loss weights.
* **`dataset.py` & `data_loader.py`**: Contains the `MultimodalManifestDataset`, `MOSEIPklDataset` (Zenodo pkl), and custom `collate_fn` for variable-length padding, attention masking, and dynamic foundation model feature extraction.
* **`model.py`**: The core `AVTCAModel` housing the projection layers, bidirectional cross-attention modules, and temporal pooling prediction heads.
* **`datamodule.py`**: The PyTorch Lightning DataModule wrapper for streamlined train, validation, and test split orchestration (manifest or Zenodo pkl).
* **`lightning_module.py`**: The training orchestrator containing the SOTA `torchmetrics`, imbalance-aware regression losses, and `SequentialLR` (Linear Warmup + Cosine Annealing) optimizers.
* **`train.py`**: The CLI-configurable entry point that pieces the modules together and launches the PyTorch Lightning `Trainer`.
* **`scripts/download_zenodo_mosei_pkl.py`**: Downloads the pre-extracted MOSEI pkl from Zenodo (COVAREP 74-dim, FACET 35-dim) for training without raw audio/video.

## Installation & Setup
1. Clone this repository and navigate to the root directory.
2. (Recommended) Create a virtual environment: `python -m venv .venv && source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows).
3. Install dependencies: `pip install -r requirements.txt`
4. Have the raw CMU-MOSEI dataset available (labels table and, for training, audio/video files).

## How to Run

### Exact steps

1. **Install**
   ```bash
   cd /student/ahmedz45/csc415/Multimodal-Sentiment-Analysis
   pip install -r requirements.txt
   ```
   If you get **"Disk quota exceeded"**, use the minimal requirements (assumes PyTorch/torchvision/torchaudio are already installed, e.g. system torch 2.8):  
   `pip install -r requirements-minimal.txt`

2. **Get CMU-MOSEI data**  
   Download the dataset (e.g. from [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK)) and put it in a folder, e.g. `./data`. You need at least:
   - A **labels file** (e.g. `data/raw/mosei_labels.csv`) with columns for video/utterance IDs, speaker, times, and the 6 emotion labels.
   - **Audio and video files** that the labels refer to (paths in the manifest must point to real `.wav` / `.mp4` files).

3. **Run training** — pick one:

   - **If you already have** `data/manifests/mosei_manifest.csv` and `data/manifests/mosei_manifest_split_stats.json`:
     ```bash
     python train.py --data_dir ./data --batch_size 16 --learning_rate 1e-4 --max_epochs 50
     ```

   - **If you do not have the manifest yet** (first time, or new data):
     ```bash
     python train.py --data_dir ./data --build_manifest \
       --labels_path ./data/raw/mosei_labels.csv \
       --audio_root ./data/audio \
       --video_root ./data/video \
       --batch_size 16 --max_epochs 50
     ```
     Use your real paths: replace `./data` with wherever you put the dataset (e.g. `/student/ahmedz45/csc415/data`), and set `--labels_path`, `--audio_root`, `--video_root` to the actual locations of the labels CSV and the audio/video folders.

That’s it. Training will start and checkpoints will be saved under `./logs/` by default.

### Option: Train with Zenodo pre-extracted pkl (no raw audio/video)
If you want to train without raw CMU-MOSEI media, use the pre-extracted [Zenodo pkl](https://zenodo.org/record/17686067) (COVAREP 74-dim, FACET 35-dim, 6 emotion labels per utterance).

1. **Download the pkl** (creates `data/processed_mosei.pkl`):
   ```bash
   python scripts/download_zenodo_mosei_pkl.py
   ```
   Or specify an output path: `python scripts/download_zenodo_mosei_pkl.py --output data/processed_mosei.pkl`

2. **Train with the pkl**:
   ```bash
   python train.py --pkl_path data/processed_mosei.pkl --batch_size 8 --max_epochs 5
   ```
   For a larger model and batch on a high-memory machine:
   ```bash
   python train.py --pkl_path data/processed_mosei.pkl --high_memory --max_epochs 5
   ```
   If disk space is tight, use `--save_weights_only` so checkpoints are smaller (no optimizer state; fine for evaluation/inference).

**Note:** Use a `.gitignore` so `data/`, `*.pkl`, `logs/`, and `__pycache__/` are not committed (see "Git ignore" below).

---

**Data path:** Replace `DATA_DIR` below with your actual path (e.g. `./data`). The repo does not include the dataset.

### Option A: Train with an existing manifest
If you already have a cleaned manifest at `DATA_DIR/manifests/mosei_manifest.csv` and `DATA_DIR/manifests/mosei_manifest_split_stats.json` (e.g. from a previous `build_manifest.py` run):

```bash
python train.py --data_dir DATA_DIR --batch_size 16 --learning_rate 1e-4 --max_epochs 50
```

Example with a real path:
```bash
python train.py --data_dir ./data --batch_size 16 --learning_rate 1e-4 --max_epochs 50
```

### Option B: Build manifest then train (required if manifest is missing)
If `DATA_DIR/manifests/mosei_manifest.csv` does not exist, the script will ask for `--labels_path`. Build the manifest and train in one go:

```bash
python train.py --data_dir DATA_DIR --build_manifest \
  --labels_path DATA_DIR/raw/mosei_labels.csv \
  --metadata_path DATA_DIR/raw/mosei_metadata.csv \
  --audio_root DATA_DIR/audio \
  --video_root DATA_DIR/video \
  --batch_size 16 --max_epochs 50
```

Example: if your data is in `./data`, use `--data_dir ./data --build_manifest --labels_path ./data/raw/mosei_labels.csv` (and point `--audio_root` / `--video_root` to where your audio/video files actually live).

### Standalone manifest build (optional)
Build only the manifest (no training):

```bash
python build_manifest.py \
  --labels-path DATA_DIR/raw/mosei_labels.csv \
  --metadata-path DATA_DIR/raw/mosei_metadata.csv \
  --audio-root DATA_DIR/audio \
  --video-root DATA_DIR/video \
  --output-dir DATA_DIR/manifests \
  --seed 42
```

Then run `train.py --data_dir DATA_DIR` as in Option A.

### Fine-tune foundation backbones
To unfreeze audio/video backbones (use a low learning rate and `num_workers 0`):

```bash
python train.py --data_dir DATA_DIR --no-freeze_backbones --learning_rate 1e-5 --num_workers 0
```

---

## Git ignore and pushing progress
Create a `.gitignore` in the repo root with at least: `data/`, `*.pkl`, `