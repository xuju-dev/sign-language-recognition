import yaml
import hashlib
import random
import numpy as np
import torch
import os
import secrets
from datetime import datetime
import time
from contextlib import contextmanager

def load_config(path: str):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def compute_config_hash(config: dict) -> str:
    """
    Compute a short hash for a config dict (used as experiment ID).
    """
    config_bytes = yaml.dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha1(config_bytes).hexdigest()[:8]


def generate_global_seed(num: int = 5):
    """
    Generate a list of strong random seeds by choosing a random number then hashing it and taking it as input to random generator.
    """
    seeds = []
    MAX32 = 2**32 - 1  # max seed range: 32 bit

    for _ in range(num):
        # 1. generate random number (secure)
        random_num = os.urandom(16)  # 128 bits

        # 2. MD5 hash it
        hash_digest = hashlib.md5(random_num).hexdigest()

        # 3. convert hex digest → integer
        seed = int(hash_digest, 16) % MAX32

        seeds.append(seed)

    return seeds


def set_global_seed(seed: int):
    """
    Make the run deterministic given a specific seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_dir(base_dir: str, model_name: str, dt: str, activation_variant: str, seed: int):
    """
    Create and return a unique output directory path.
    """
    model_dir_name = model_name.split("/")[-1]
    model_dir_name = model_dir_name.split(".")[0]
    run_dir = os.path.join(base_dir, f"{model_dir_name}", f"{activation_variant}", f"{dt}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_metadata(run_dir: str, cfg: dict, seed: int, activation_variant):
    """
    Save configuration, seed, and hash info for full reproducibility.
    """
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": seed,
        "activation_variant": activation_variant,
        "config": cfg,
    }
    with open(os.path.join(run_dir, "metadata.yaml"), "w") as f:
        yaml.dump(metadata, f)
    return metadata

def time_block(label: str, func, *args, log_path: str = None, **kwargs):
    """
    Measure how long a function takes to run, print duration, and optionally log to file.

    Usage:
        result, duration = time_block("Training", trainer.train)
        result, duration = time_block("Validation", validate, val_loader, log_path="timelog.json")
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start

    print(f"⏱️ {label} took {duration:.2f} seconds ({duration/60:.2f} minutes)")

    return result, duration

