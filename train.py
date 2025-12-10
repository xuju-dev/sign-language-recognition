import os
import json
import torch
import numpy as np
from datetime import datetime

from sklearn.model_selection import RepeatedStratifiedKFold
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
    rkf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed
    )
    total_folds = n_splits * n_repeats
    print(f"Using {n_splits}-fold CV repeated {n_repeats} times "
          f"({total_folds} total folds).")

    print("\nLOADING DEVICE...\n")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------
    # 5. Output directory & metadata
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = prepare_output_dir(cfg["output_dir"], cfg["model_name"], dt, activation_variant=activation_variant, seed=seed)
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
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=seed,
    )

    # -----------------------
    # 7. Repeated K-fold loop
    # -----------------------
    fold_metrics = []
    fold_idx = 0

    print("\nSTARTING REPEATED K-FOLD CV...\n")

    for train_indices, val_indices in rkf.split(np.arange(num_samples), all_labels):
        fold_idx += 1
        print(f"\n========== Fold {fold_idx}/{total_folds} ==========")

        train_subset = Subset(dev_dataset, train_indices)
        val_subset = Subset(dev_dataset, val_indices)

        # Fresh model for each fold
        model = load_mobilev3_model(
            activation_variant=activation_variant,
            num_labels=num_labels,
            device=device,
        )

        # Per-fold output dir
        fold_output_dir = os.path.join(run_dir, f"fold_{fold_idx}")

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

        print("\nTRAINING FOLD...\n")
        _, train_duration = time_block(f"Training fold {fold_idx}", trainer.train)

        fold_eval_metrics = trainer.evaluate()
        fold_eval_metrics["training_duration_minutes"] = float(train_duration / 60.0)
        print(f"Fold {fold_idx} metrics: {fold_eval_metrics}")

        fold_metrics.append(fold_eval_metrics)

    # -----------------------
    # 8. Aggregate CV metrics
    # -----------------------
    # Compute mean/std for numeric eval_* metrics
    aggregated_metrics = {}
    if fold_metrics:
        # Get keys that look like eval_* and are numeric in the first fold
        for key, value in fold_metrics[0].items():
            if key.startswith("eval_") and isinstance(value, (int, float)):
                values = [fm[key] for fm in fold_metrics]
                aggregated_metrics[f"cv_mean_{key}"] = float(np.mean(values))
                aggregated_metrics[f"cv_std_{key}"] = float(np.std(values))

    print("\n===== CROSS-VALIDATION SUMMARY =====")
    for k, v in aggregated_metrics.items():
        print(f"{k}: {v:.4f}")

    # Also keep raw per-fold metrics
    results = {
        "fold_metrics": fold_metrics,
        "cv_summary": aggregated_metrics,
    }

    print("\nSAVING EXPERIMENT...\n")
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"Repeated CV complete! Results stored in {run_dir}")


if __name__ == "__main__":
    CONFIG_PATH = "configs/mobilenet_v3.yaml"
    model_variants = ['original']  # or ['original', 'relu', 'leakyrelu']

    global_seeds = generate_global_seed()  # list[int]
    print(f"Generated global seeds: {global_seeds}")

    for model in model_variants:
        for g_seed in global_seeds:
            print(f"\n\n### Running experiment with global seed {g_seed}, variant {model} ###\n")
            run_experiment(CONFIG_PATH, g_seed, activation_variant=model)
