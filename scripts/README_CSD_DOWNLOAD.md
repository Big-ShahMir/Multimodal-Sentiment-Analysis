# Which .csd files to download for CMU-MOSEI

Download **only these** (no language folder) so the subset fits and matches the audio+video pipeline:

| Source | File | Size | URL |
|--------|------|------|-----|
| **labels** | `CMU_MOSEI_Labels.csd` | 22 MB | http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/ |
| **acoustic** | `CMU_MOSEI_COVAREP.csd` | 11 GB | http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/acoustic/ |
| **visual** | `CMU_MOSEI_VisualFacet42.csd` | 1.5 GB | http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/visual/ |

**Do not download** (saves ~17 GB and we don't use them):
- `CMU_MOSEI_VisualOpenFace2.csd` (16 GB)
- Everything in `language/` (TimestampedWords, TimestampedWordVectors, TimestampedPhones)

**Total:** ~12.5 GB. Put all three files in one folder, e.g. `data/cmu_mosei_csd/`.

Install the CMU Multimodal SDK (on the machine where the .csd files are):
```bash
pip install mmsdk
# or: pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
```

Then run the balanced preprocessing script from the repo root:
```bash
python scripts/build_mosei_balanced_pkl.py --csd_dir data/cmu_mosei_csd --output_path data/mosei_balanced.pkl --samples_per_emotion 500
```
This samples **the same number** of segments per emotion (500 per emotion = 3000 total by default), then splits into train/val/test and saves one `.pkl`. Copy the `.pkl` to the lab PC and use a dataset loader that reads it (we can add that to the repo).
