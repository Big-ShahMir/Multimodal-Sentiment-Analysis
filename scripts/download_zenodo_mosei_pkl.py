#!/usr/bin/env python3
"""
Download processed_mosei.pkl from Zenodo (record 17686067), same as in the original
Research Project notebook. Saves to data/ by default so train.py --pkl_path can use it.

When you train with --pkl_path, the pipeline uses these pre-extracted features
(COVAREP 74-dim, FACET 35-dim) directly. It does *not* run Wav2Vec2 or ResNet/ViT;
the AVT-CA model is still the same (fusion/attention), but inputs are from the pkl,
not from foundation-model extractors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

ZENODO_RECORD_ID = "17686067"
DEFAULT_OUTPUT = Path("data") / "processed_mosei.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download processed_mosei.pkl from Zenodo (record 17686067)."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path for the .pkl file (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--record_id",
        type=str,
        default=ZENODO_RECORD_ID,
        help=f"Zenodo record ID (default: {ZENODO_RECORD_ID}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output file already exists.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"{output} already exists. Use --force to re-download.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)

    api_url = f"https://zenodo.org/api/records/{args.record_id}"
    print(f"Fetching Zenodo record {args.record_id}...")
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    record = resp.json()

    pkl_file = None
    for f in record.get("files", []):
        key = f.get("key", "")
        if key.endswith(".pkl"):
            pkl_file = f
            break

    if not pkl_file:
        print("No .pkl file found in this Zenodo record.", file=sys.stderr)
        sys.exit(1)

    url = pkl_file["links"]["self"]
    size = pkl_file.get("size", 0)
    key = pkl_file.get("key", "file.pkl")
    print(f"Downloading {key} ({size:,} bytes) to {output}...")

    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0

    with open(output, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = 100.0 * downloaded / total
                    print(f"\rProgress: {pct:.1f}% ({downloaded:,}/{total:,} bytes)", end="", flush=True)
    print(f"\nDone. Saved to {output}")
    print("Run training with: python train.py --pkl_path", output)


if __name__ == "__main__":
    main()
