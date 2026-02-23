import sys
import os

# Add the parent directory to sys.path so 'mmsdk' (which lives in dataset download/) is importable
sdk_path = os.path.abspath("..")
sys.path.insert(0, sdk_path)

import pandas as pd
from mmsdk import mmdatasdk

def generate_manifest(csd_path, output_csv):
    print("Loading Kaggle CSD file (this might take a moment)...")
    comp_seq = mmdatasdk.computational_sequence(csd_path)
    
    rows = []
    # Loop through the raw data dictionary
    for unique_id, data in comp_seq.data.items():
        # CMU keys usually look like "VideoID[utterance_index]"
        video_id = unique_id.split('[')[0] if '[' in unique_id else unique_id
        
        intervals = data['intervals']
        features = data['features']
        
        # Match each time interval with its corresponding label features
        for i in range(len(intervals)):
            rows.append({
                'video_id': video_id,
                'utterance_id': unique_id,
                'start_time': intervals[i][0],
                'end_time': intervals[i][1],
                'raw_features': features[i].tolist()            })

    # Convert to Pandas and export
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Success! {len(df)} utterances saved to {output_csv}.")

if __name__ == "__main__":
    kaggle_csd_file = "CMU_MOSEI_Labels.csd"  # File is in the same data/ folder
    generate_manifest(kaggle_csd_file, "full_manifest.csv")