import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"

def extract_latest_jsons_per_model(root_dir):
    root_dir = Path(root_dir)
    model_dfs = {}

    for model_dir in root_dir.iterdir():
        if not model_dir.is_dir():
            continue

        # parse date folders safely
        dated_dirs = []
        for d in model_dir.iterdir():
            if not d.is_dir():
                continue
            try:
                ts = datetime.strptime(d.name, TIMESTAMP_FMT)
                dated_dirs.append((ts, d))
            except ValueError:
                continue  # skip non-date folders

        if not dated_dirs:
            continue

        # select latest timestamp
        latest_ts, latest_dir = max(dated_dirs, key=lambda x: x[0])

        records = []
        print(f"Extracting from model '{model_dir.name}' at timestamp '{latest_ts}'")
        for fold_dir in latest_dir.iterdir():
            if not fold_dir.is_dir():
                continue

            for json_file in fold_dir.glob("*.json"):
                with open(json_file, "r") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    row = pd.json_normalize(data).iloc[0].to_dict()
                    records.append({
                        **row,
                        "model_variant": model_dir.name,
                        "timestamp": latest_ts,
                        "fold": fold_dir.name
                    })

                elif isinstance(data, list):
                    for item in data:
                        row = pd.json_normalize(item).iloc[0].to_dict()
                        records.append({
                            **row,
                            "model_variant": model_dir.name,
                            "timestamp": latest_ts,
                            "fold": fold_dir.name
                        })

        if records:
            model_dfs[model_dir.name] = pd.DataFrame(records)

    return model_dfs

if __name__ == "__main__":
    root_directory = "./experiments"
    output_csv = "compiled_experimental_results.csv"

    model_dataframes = extract_latest_jsons_per_model(root_directory)

    all_dfs = []
    for model_name, df in model_dataframes.items():
        all_dfs.append(df)

    if all_dfs:
        min_accs = [df['eval_accuracy'].min() for df in all_dfs if 'eval_accuracy' in df.columns]
        print("Minimum accuracies per model variant:", min_accs)
        max_accs = [df['eval_accuracy'].max() for df in all_dfs if 'eval_accuracy' in df.columns]
        print("Maximum accuracies per model variant:", max_accs)
        acc_ranges = np.mean(np.array(max_accs) - np.array(min_accs))
        print("Averaged accuracy ranges per model variant:", acc_ranges)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_csv, index=False)
        print(f"Compiled results saved to {output_csv}")
    else:
        print("No experimental results found.")