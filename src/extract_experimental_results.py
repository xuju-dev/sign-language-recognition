import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"

def extract_latest_jsons_per_model(root_dir):
    root_dir = Path(root_dir)
    model_dfs = {}
    repeat_results_df = {}

    for model_dir in root_dir.iterdir():
        if not model_dir.is_dir():
            continue

        records = []
        for seed_dir in model_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            print(f"Extracting from model '{model_dir.name}' with seed '{seed_dir}'")
            for json_file in seed_dir.glob("*.json"):
                with open(json_file, "r") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    print("Dict")
                    row = pd.json_normalize(data).iloc[0].to_dict()
                    meta_data = row.copy().pop('repeat_results')
                    repeat_results = row['repeat_results'][0]['repeat_metrics'][0]
                    # Extract repeat results
                    acc = repeat_results['eval_accuracy']
                    print(acc)
                    print(repeat_results)
                    repeat_results_dict = {idx: val for idx, val in enumerate(repeat_results)}
                    records.append({
                        "model_variant": model_dir.name,
                        "global seed": seed_dir.name,
                        'eval_accuracy': acc,
                        'eval_macro_f1': repeat_results['eval_macro_f1'],
                        'fold_training_duration_minutes': repeat_results['fold_training_duration_minutes'],
                        **meta_data[0],
                        # **repeat_results_dict,
                    })

                elif isinstance(data, list):
                    print("List")
                    for item in data:
                        row = pd.json_normalize(item).iloc[0].to_dict()
                        meta_data = row.copy()[:-1]
                        meta_data_dict = {k: v for k, v in enumerate(meta_data)}
                        print(meta_data_dict)
                        records.append({
                            "model_variant": model_dir.name,
                            "global seed": seed_dir.name,
                            **meta_data_dict,
                        })

        if records:
            model_dfs[model_dir.name] = pd.DataFrame(records)
            # repeat_metrics = model_dfs['repeat_metrics'][0]
            # repeat_results_df[model_dir.name] = pd.DataFrame(repeat_metrics)

    return model_dfs


def extract_metric_vectors_per_model(root_dir, repeats_per_seed=5, folds_per_repeat=5):
    """For each model variant and seed, return 25-entry vectors for accuracy, f1, and duration.

    Assumes JSON structure where `repeat_results` is a list of repeats and each repeat
    contains `repeat_metrics` which is a list of per-fold dicts containing
    `eval_accuracy`, `eval_macro_f1`, and `fold_training_duration_minutes`.
    """
    root_dir = Path(root_dir)
    rows = []

    for model_dir in root_dir.iterdir():
        if not model_dir.is_dir():
            continue

        for seed_dir in model_dir.iterdir():
            if not seed_dir.is_dir():
                continue

            # find a json file (prefer final_results.json)
            json_path = seed_dir / "final_results.json"
            if not json_path.exists():
                json_files = list(seed_dir.glob("*.json"))
                json_path = json_files[0] if json_files else None

            if json_path is None:
                continue

            with open(json_path, "r") as f:
                data = json.load(f)

            # collect metrics in order: repeat 0..R-1, within each repeat folds 0..F-1
            accs = []
            f1s = []
            durs = []

            repeat_results = data['repeat_results']

            for rep in repeat_results:
                accs.append(rep.get("mean_accuracy"))
                f1s.append(rep.get("mean_macro_f1"))
                durs.append(rep.get("repeat_training_duration_minutes"))

            # If there are fewer entries than expected, pad with NaN; if more, truncate
            expected = repeats_per_seed
            def fit_list(lst):
                arr = list(lst)[:expected]
                if len(arr) < expected:
                    arr.extend([np.nan] * (expected - len(arr)))
                return arr

            accs = fit_list(accs)
            f1s = fit_list(f1s)
            durs = fit_list(durs)

            rows.append({
                "model_variant": model_dir.name,
                "global_seed": seed_dir.name,
                "accuracy_vector": accs,
                "f1_vector": f1s,
                "duration_vector_minutes": durs,
            })

    return pd.DataFrame(rows)


def get_metric_matrices(root_dir, repeats_per_seed=5):
    """Return per-model numpy matrices for accuracy, f1 and duration.

    Returns a dict mapping model_variant -> {
        'accuracy': ndarray shape (n_seeds, repeats_per_seed),
        'f1': ndarray same shape,
        'duration': ndarray same shape,
        'seeds': list of seed identifiers in the same row order
    }
    """
    df = extract_metric_vectors_per_model(root_dir, repeats_per_seed=repeats_per_seed)
    results = {}
    if df.empty:
        return results

    for model_variant, group in df.groupby('model_variant'):
        seeds = list(group['global_seed'])
        acc_matrix = np.vstack(group['accuracy_vector'].tolist()) if not group['accuracy_vector'].isnull().all() else np.empty((0, repeats_per_seed))
        f1_matrix = np.vstack(group['f1_vector'].tolist()) if not group['f1_vector'].isnull().all() else np.empty((0, repeats_per_seed))
        dur_matrix = np.vstack(group['duration_vector_minutes'].tolist()) if not group['duration_vector_minutes'].isnull().all() else np.empty((0, repeats_per_seed))

        results[model_variant] = {
            'accuracy': acc_matrix,
            'f1': f1_matrix,
            'duration': dur_matrix,
            'seeds': seeds,
        }

    return results


def get_flat_metric_vectors(root_dir, repeats_per_seed=5):
    """Return flattened vectors per model: n_seeds * repeats_per_seed entries.

    The order is seed-major: [seed0.rep0..repN, seed1.rep0..repN, ...]
    """
    mats = get_metric_matrices(root_dir, repeats_per_seed=repeats_per_seed)
    flat = {}
    for model, d in mats.items():
        acc = d['accuracy']
        f1 = d['f1']
        dur = d['duration']
        # flatten row-major (seed-major)
        acc_flat = acc.flatten() if acc.size else np.array([])
        f1_flat = f1.flatten() if f1.size else np.array([])
        dur_flat = dur.flatten() if dur.size else np.array([])
        flat[model] = {
            'accuracy': acc_flat,
            'f1': f1_flat,
            'duration': dur_flat,
            'seeds': d['seeds'],
        }
    return flat

if __name__ == "__main__":
    root_directory = "./output"
    # Save legacy compiled results (if any) and the new metric vectors
    legacy_output_csv = "./output/simple_experimental_results_0.1-5.csv"

    # Existing behavior: keep producing the combined eval rows (best-effort)
    model_dataframes = extract_latest_jsons_per_model(root_directory)
    all_dfs = [df for df in model_dataframes.values()]
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        try:
            combined_df.to_csv(legacy_output_csv, index=False)
            print(f"Compiled legacy results saved to {legacy_output_csv}")
        except Exception:
            print("Could not save legacy combined CSV.")

    # New behavior: extract 25-entry vectors per (model_variant, global_seed)
    vectors_df = extract_metric_vectors_per_model(root_directory)
    metrics_output_csv = "./output/simple_experimental_results_metrics.csv"
    if not vectors_df.empty:
        # Expand vectors into separate columns for clarity (acc_0..acc_24 etc.)
        expected = 25
        acc_cols = [f"acc_{i}" for i in range(expected)]
        f1_cols = [f"f1_{i}" for i in range(expected)]
        dur_cols = [f"dur_{i}" for i in range(expected)]

        expanded = pd.DataFrame()
        expanded[["model_variant", "global_seed"]] = vectors_df[["model_variant", "global_seed"]]

        acc_matrix = pd.DataFrame(vectors_df["accuracy_vector"].tolist(), columns=acc_cols)
        f1_matrix = pd.DataFrame(vectors_df["f1_vector"].tolist(), columns=f1_cols)
        dur_matrix = pd.DataFrame(vectors_df["duration_vector_minutes"].tolist(), columns=dur_cols)

        expanded = pd.concat([expanded, acc_matrix, f1_matrix, dur_matrix], axis=1)
        expanded.to_csv(metrics_output_csv, index=False)
        print(f"Saved metric vectors to {metrics_output_csv}")
    else:
        print("No metric vectors extracted.")