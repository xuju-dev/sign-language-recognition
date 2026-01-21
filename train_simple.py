"""
Training script for CNN models variant with different layer structures
using 5 times repeated 5-fold cross-validation.
"""

import os
import json
from pathlib import Path
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from src.utils import generate_global_seed, set_global_seed
from src.dataloader.dataloader_simple import load_datasets
from src.models.simple_cnn import BaselineCNN, DeeperCNN, RegularizedCNN


def run_experiment(dev_dataset, labels, seed, model_class, num_classes=28, num_repeats=5, num_folds=5,
                   epochs=2, batch_size=32, lr=1e-3, output_dir="output"):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    # Prepare labels for stratification
    num_samples = len(dev_dataset)
    all_labels = np.array([dev_dataset[i][1] for i in range(len(dev_dataset))])

    repeat_results = []

    exp_start = time.time()

    for repeat_idx in range(num_repeats):
        print(f"\n===== REPEAT {repeat_idx+1}/{num_repeats} =====")
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed + repeat_idx)
        fold_metrics = []

        repeat_start = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(num_samples), all_labels)):
            print(f"\n--- [{seed}] Fold {fold_idx+1}/{num_folds} ---")
            fold_output_dir = os.path.join(output_dir, f"repeat_{repeat_idx+1}", f"fold_{fold_idx+1}")
            repeat_output_dir = os.path.join(output_dir, f"repeat_{repeat_idx+1}")
            os.makedirs(fold_output_dir, exist_ok=True)
            checkpoint_path = os.path.join(fold_output_dir, "checkpoint.pth")

            train_subset = Subset(dev_dataset, train_idx)
            val_subset = Subset(dev_dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size)

            # Model
            model = model_class[1](num_classes=num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Track cumulative training time for resuming
            training_info_path = os.path.join(fold_output_dir, "training_info.json")

            if os.path.exists(training_info_path):
                with open(training_info_path, "r") as f:
                    training_info = json.load(f)
            else:
                training_info = {"completed": False, "cumulative_training_minutes": 0}

            previous_duration = training_info["cumulative_training_minutes"]
            
            # Resume from checkpoint if exists
            start_epoch = 0
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                start_epoch = ckpt["epoch"] + 1
                print(f"↪️ Resuming from epoch {start_epoch}")

            fold_start = time.time()

            # Training loop
            for epoch in range(start_epoch, epochs):
                model.train()
                running_loss = 0.0
                for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                avg_loss = running_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

                # Save checkpoint at each epoch
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict()
                }, checkpoint_path)

            fold_end = time.time()
            incremental_duration = (fold_end - fold_start) / 60

            # Update cumulative duration
            fold_duration_minutes = previous_duration + incremental_duration
            training_info["cumulative_training_minutes"] = fold_duration_minutes

            # Check if training completed for this fold
            training_info["completed"] = start_epoch >= epochs
            
            # Save cumulative training info
            with open(training_info_path, "w") as f:
                json.dump(training_info, f, indent=4)

            # Evaluation
            if training_info["completed"]:
                print("Fold completed.")
            
            model.eval()

            all_preds = []
            all_gt = []

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Evaluating", leave=False):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_gt.extend(labels.cpu().numpy())

            acc = np.mean(np.array(all_preds) == np.array(all_gt))
            assert len(all_preds) == len(all_gt) == len(val_subset), \
                f"Length mismatch: preds={len(all_preds)}, labels={len(all_gt)}, val_subset={len(val_subset)}"

            fold_f1 = f1_score(all_gt, all_preds, average='macro')

            fold_duration = training_info["cumulative_training_minutes"]
            fold_metrics.append({
                "eval_accuracy": acc,
                "eval_macro_f1": fold_f1,
                "fold_training_duration_minutes": fold_duration
            })

            # Save fold results
            with open(os.path.join(fold_output_dir, "results.json"), "w") as f:
                json.dump(fold_metrics, f, indent=4)

            print(f"Fold {fold_idx+1} accuracy: {acc:.3f}, duration: {fold_duration:.2f} min")

        fold_cumulative_durations = []
        for file in Path(repeat_output_dir).rglob("fold_*/training_info.json"):
            with open(file, "r") as f:
                fold_cumulative_durations.append(json.load(f)["cumulative_training_minutes"])
        
        repeat_duration = sum(fold_cumulative_durations)
        repeat_summary = {
            "repeat_training_duration_minutes": repeat_duration,
            "mean_accuracy": float(np.mean([f["eval_accuracy"] for f in fold_metrics])),
            "mean_macro_f1": float(np.mean([f["eval_macro_f1"] for f in fold_metrics])),
            "repeat_metrics": fold_metrics
        }
        repeat_results.append(repeat_summary)

        repeat_output_dir = os.path.join(output_dir, f"repeat_{repeat_idx+1}")
        with open(os.path.join(repeat_output_dir, "intermediate_results.json"), "w") as f:
            json.dump(repeat_results, f, indent=4)

        print(f"\n[SEED: {seed}|MODEL: {model_class[0]}] Repeat {repeat_idx+1} mean accuracy: {repeat_summary['mean_accuracy']:.3f}, duration: {repeat_duration:.2f} min")

    intermediate_results = []
    repeat_cumulative_durations = []
    for file in Path(output_dir).rglob("intermediate_results.json"):
        with open(file, "r") as f:
            results = json.load(f)
            repeat_cumulative_durations.append(results[0]["repeat_training_duration_minutes"])
            intermediate_results.append(results)

    print(f"Total experiment duration: {sum(repeat_cumulative_durations):.2f} minutes")

    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump({
            "model_variant": model_class[0],
            "global_seed": seed,
            "num_repeats": num_repeats,
            "experiment_duration_minutes": sum(repeat_cumulative_durations),
            "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp_start)),
            "repeat_results": repeat_results,
            # "intermediate_results": intermediate_results,
            }, 
            f, 
            indent=4
        )
    


if __name__ == "__main__":
    SEED_FILE = "output/global_seeds.json"
    NUM_SEEDS = 5

    # Load or generate seeds
    if os.path.exists(SEED_FILE):
        with open(SEED_FILE, "r") as f:
            global_seeds = json.load(f)
        print(f"Loaded existing global seeds: {global_seeds}\n")
    else:
        global_seeds = generate_global_seed(num=NUM_SEEDS)
        with open(SEED_FILE, "w") as f:
            json.dump(global_seeds, f)
        print(f"Generated and saved global seeds: {global_seeds}\n")

    # Load datasets
    train_dataset, val_dataset, test_dataset, labels, _ = load_datasets()
    dev_dataset = ConcatDataset([train_dataset, val_dataset])

    model_variants = {
        "BaselineCNN": BaselineCNN,
        "DeeperCNN": DeeperCNN,
        "RegularizedCNN": RegularizedCNN
    }

    for g_seed in global_seeds:
        for model in model_variants.items():
            print(f"\n=== Running experiment with model {model[0]} [Global Seed: {g_seed}] ===\n")
            set_global_seed(g_seed)
            run_experiment(
                dev_dataset=dev_dataset,
                labels=labels,
                seed=g_seed,
                model_class=model,
                num_repeats=5,
                num_folds=5,
                epochs=5,
                batch_size=32,
                lr=1e-3,
                output_dir=f"output/{model[0]}/{g_seed}"
            )
