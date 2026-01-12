import os
import json
import time
from pathlib import Path

import torch
import numpy as np
from datetime import datetime

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, ConcatDataset

from transformers import TrainingArguments, Trainer
from evaluate import load as load_metric

from src.dataloader.dataloader import load_datasets
from src.models.mobilenetv3 import load_mobilev3_model
from src.utils import (
    load_config,
    generate_global_seed,
    set_global_seed,
    prepare_output_dir,
    save_metadata,
    time_block,
)

def run_experiment(config_path: str, seed, activation_variant: str = 'original'):
    print("SETTING UP THE EXPERIMENT...\n")

    # Make sure seed is a plain int
    seed = int(seed)

    # -----------------------
    # 0. Start full experiment timer
    # -----------------------
    exp_start = time.time()

    # -----------------------
    # 1. Load config and hash
    # -----------------------
    cfg = load_config(config_path)
    print(f"Loaded config: {config_path}")
    seed_group = f'{activation_variant}_{seed}'

    # -----------------------
    # 2. Random seed
    # -----------------------
    print(f"Random seed: {seed}")
    set_global_seed(seed)

    print("\nLOADING DATA...\n")
    # -----------------------
    # 3. Load data
    # -----------------------
    train_dataset, val_dataset, _, labels = load_datasets()
    num_labels = len(labels)

    # Build dev_dataset for CV: use all available labeled data
    # If you prefer just train_dataset, replace with dev_dataset = train_dataset
    dev_dataset = ConcatDataset([train_dataset, val_dataset])
    num_samples = len(dev_dataset)
    print(f"Dev dataset size (for CV): {num_samples}")

    # Extract labels for stratification
    # Assumes each item in dev_dataset returns a dict with key "labels"
    all_labels = []
    for i in range(num_samples):
        item = dev_dataset[i]
        # Adjust key name if your dataset uses something else (e.g. "label")
        all_labels.append(int(item["labels"]))
    all_labels = np.array(all_labels)

    # -----------------------
    # 4. CV setup: 5-fold, 5 repeats
    # -----------------------
    n_splits = 5
    n_repeats = 5

    print(f"Using {n_splits}-fold CV repeated {n_repeats} times (with seed {seed})")

    print("\nLOADING DEVICE...\n")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------
    # 5. Output directory & metadata
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = prepare_output_dir(cfg["output_dir"], cfg["model_name"], activation_variant=activation_variant, seed=seed)
    save_metadata(run_dir, cfg, seed, activation_variant)
    print(f"Saved model metadata to directory: {run_dir}")

    # -----------------------
    # 6. Metrics
    # -----------------------
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = accuracy_metric.compute(predictions=preds, references=p.label_ids)
        f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="weighted")
        return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

    # -----------------------
    # Base TrainingArguments (cloned per fold)
    # -----------------------
    base_training_args = dict(
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        learning_rate=float(cfg["lr"]),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["num_train_epochs"],
        logging_steps=cfg["logging_steps"],
        load_best_model_at_end=True,
        weight_decay=0.01,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=seed,
        dataloader_pin_memory=False,  # MPS compatibility, uncomment if CUDA is used
    )

    # -----------------------
    # 7. Repeated K-fold loop (TRAINING)
    # -----------------------
    repeat_results = []

    for repeat_idx in range(n_repeats):
        print(f"\n===== REPEAT {repeat_idx + 1}/{n_repeats} =====")

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + repeat_idx,
        )

        fold_metrics = []
        fold_idx = 0

        # Start repeat timer
        repeat_start = time.time()

        for train_idx, val_idx in skf.split(np.arange(num_samples), all_labels):
            fold_idx += 1
            print(f"\n--- ({repeat_idx + 1}) Fold {fold_idx}/{n_splits} ---")

            train_subset = Subset(dev_dataset, train_idx)
            val_subset = Subset(dev_dataset, val_idx)

            model = load_mobilev3_model(
                activation_variant=activation_variant,
                num_labels=num_labels,
                device=device,
            )

            fold_output_dir = os.path.join(
                run_dir, f"repeat_{repeat_idx + 1}", f"fold_{fold_idx}"
            )

            training_args = TrainingArguments(
                output_dir=fold_output_dir,
                logging_dir=os.path.join(fold_output_dir, "logs"),
                **base_training_args,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_subset,
                eval_dataset=val_subset,
                compute_metrics=compute_metrics,
            )

            # Check if any checkpoint exists in the fold output dir to resume to previous training
            checkpoint_dirs = [
                os.path.join(fold_output_dir, d)
                for d in os.listdir(fold_output_dir) if d.startswith("checkpoint-")
            ]
            
            # Track cumulative training time for resuming
            training_info_path = os.path.join(fold_output_dir, "training_info.json")

            if os.path.exists(training_info_path):
                with open(training_info_path, "r") as f:
                    training_info = json.load(f)
            else:
                training_info = {"completed": False, "cumulative_training_minutes": 0}

            previous_duration = training_info["cumulative_training_minutes"]

            # Start fold timer
            fold_start = time.time()

            if checkpoint_dirs:
                # Resume from the latest checkpoint
                latest_ckpt = max(checkpoint_dirs, key=os.path.getctime)
                print(f"\n↪️ Resuming from checkpoint: {latest_ckpt}\n")
                trainer.train(resume_from_checkpoint=latest_ckpt)
            else:
                # Start from scratch
                trainer.train()

            fold_end = time.time()

            incremental_duration = (fold_end - fold_start) / 60

            # Update cumulative duration
            fold_duration_minutes = previous_duration + incremental_duration
            training_info["cumulative_training_minutes"] = fold_duration_minutes

            # Check if training completed for this fold
            if trainer.state.global_step >= trainer.state.max_steps:
                training_info["completed"] = True
            
            # Save cumulative training info
            with open(training_info_path, "w") as f:
                json.dump(training_info, f, indent=4)

            # EVALUATION
            metrics = trainer.evaluate()
            metrics["training_duration_minutes"] = training_info["cumulative_training_minutes"]
            fold_metrics.append(metrics)
            print(f"\nFold {fold_idx} evaluation metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v}")
            
            # Save intermediate results after one fold
            with open(os.path.join(fold_output_dir, "results.json"), "w") as f:
                json.dump(fold_metrics, f, indent=4)

        # aggregate one 5-fold CV → one value
        repeat_summary = {}

        repeat_output_dir = os.path.join(
            run_dir, f"repeat_{repeat_idx + 1}"
        )

        # Get fold train durations
        fold_cumulative_durations = []
        for file in Path(repeat_output_dir).rglob("fold_*/training_info.json"):
            with open(file, "r") as f:
                fold_cumulative_durations.append(json.load(f)["cumulative_training_minutes"])
        
        repeat_summary["repeat_duration_minutes"] = sum(fold_cumulative_durations)

        for key, val in fold_metrics[0].items():
            if key.startswith("eval_") and isinstance(val, (int, float)):
                repeat_summary[key] = float(np.mean([fm[key] for fm in fold_metrics]))

        repeat_results.append(repeat_summary)
        print(f"\nRepeat {repeat_idx + 1} CV summary metrics:")
        for k, v in repeat_summary.items():
            print(f"{k}: {v}")

        # Save intermediate results after all repeats
        with open(os.path.join(repeat_output_dir, "results.json"), "w") as f:
            json.dump(repeat_results, f, indent=4)
    
    # -----------------------
    # 8. Aggregate CV metrics
    # -----------------------
    final_results = {}
    final_results["model_variant"] = activation_variant
    final_results["global_seed"] = seed
    final_results["training_datetime"] = dt

    for key in repeat_results[0]:
        final_results[key] = [r[key] for r in repeat_results]

    # Get fold train durations
    repeat_cumulative_durations = []
    for file in Path(repeat_output_dir).rglob("result.json"):
        with open(file, "r") as f:
            fold_cumulative_durations.append(json.load(f)["repeat_duration_minutes"])

    print(f"\nTotal experiment duration for seed {g_seed}, model {model}: {sum(repeat_cumulative_durations):.2f} minutes")
    final_results["experiment_duration_minutes"] = sum(repeat_cumulative_durations)

    with open(os.path.join(run_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n===== FINAL RESULTS =====")
    for k, v in final_results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    CONFIG_PATH = "configs/mobilenet_v3.yaml"
    SEED_FILE = "experiments/global_seeds.json"
    NUM_SEEDS = 5

    # Load or generate seeds
    if os.path.exists(SEED_FILE):
        with open(SEED_FILE, "r") as f:
            global_seeds = json.load(f)
        print(f"Loaded existing global seeds: {global_seeds}")
    else:
        global_seeds = generate_global_seed(num=NUM_SEEDS)
        with open(SEED_FILE, "w") as f:
            json.dump(global_seeds, f)
        print(f"Generated and saved global seeds: {global_seeds}")

    if os.path.exists(CONFIG_PATH):
        print(f"Using config file at: {CONFIG_PATH}")

    model_variants = ['original', 'relu', 'leakyrelu']  # or ['original', 'relu', 'leakyrelu']

    for g_seed in global_seeds:
        for model in model_variants:
            print(f"\n\n### Running experiment with global seed {g_seed}, variant {model} ###\n")
            run_experiment(CONFIG_PATH, g_seed, activation_variant=model)

