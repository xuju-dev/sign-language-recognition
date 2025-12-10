import os
import random
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

def preprocess_dataset(root, split_ratio, subset_fraction, img_resize, seed):
    random.seed(seed)
    np.random.seed(seed)

    preprocessed_dir = os.path.join(root, f'asl_alphabet_preprocessed_{img_resize}_{subset_fraction}')
    os.makedirs(preprocessed_dir, exist_ok=True)

    # === TRAIN / VAL SPLIT ===
    train_dir = os.path.join(root, 'train')
    train_pre = os.path.join(preprocessed_dir, 'train')
    val_pre = os.path.join(preprocessed_dir, 'val')

    os.makedirs(train_pre, exist_ok=True)
    os.makedirs(val_pre, exist_ok=True)

    total_images = sum(len(files) for _, _, files in os.walk(train_dir))

    with tqdm(total=max(1, (total_images * subset_fraction)), desc="Preprocessing train/val folders", ncols=100) as pbar:
        for cls in os.listdir(train_dir):
            cls_path = os.path.join(train_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            os.makedirs(os.path.join(train_pre, cls), exist_ok=True)
            os.makedirs(os.path.join(val_pre, cls), exist_ok=True)

            imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(imgs)

            # === TAKE SUBSET ===
            if subset_fraction < 1.0:
                subset_size = max(1, int(len(imgs) * subset_fraction))
                imgs = random.sample(imgs, subset_size)

            split_idx = int(len(imgs) * split_ratio)
            train_imgs = imgs[:split_idx]
            val_imgs = imgs[split_idx:]

            for subset, img_list in [('train', train_imgs), ('val', val_imgs)]:
                for img_name in img_list:
                    src = os.path.join(cls_path, img_name)
                    dst = os.path.join(preprocessed_dir, subset, cls, img_name)

                    img = Image.open(src).convert('RGB')
                    img = img.resize((img_resize, img_resize))
                    arr = np.asarray(img) / 255.0
                    img = Image.fromarray((arr * 255).astype(np.uint8))
                    img.save(dst)
                    pbar.update(1)

    # === TEST FOLDER ===
    test_dir = os.path.join(root, 'test')
    test_pre = os.path.join(preprocessed_dir, 'test')
    os.makedirs(test_pre, exist_ok=True)

    total_images = sum(len(files) for _, _, files in os.walk(test_dir))

    with tqdm(total=total_images, desc="Preprocessing test folder", ncols=100) as pbar:
        for cls in os.listdir(test_dir):
            cls_path = os.path.join(test_dir, cls)
            if not os.path.isdir(cls_path):
                continue

            os.makedirs(os.path.join(test_pre, cls), exist_ok=True)

            for img_name in os.listdir(cls_path):
                src = os.path.join(cls_path, img_name)
                dst = os.path.join(test_pre, cls, img_name)

                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img = Image.open(src).convert('RGB')
                img = img.resize((img_resize, img_resize))
                arr = np.asarray(img) / 255.0
                img = Image.fromarray((arr * 255).astype(np.uint8))
                img.save(dst)
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset and split train/val sets.")
    parser.add_argument("--data_root", type=str, default='data/asl_alphabet_dataset', help="Root dataset directory.")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Train/validation split ratio (default: 0.8 for 80/20).")
    parser.add_argument("--img_resize", type=int, default=224, help="Image size after resizing (default: 224 for 224x224)")
    parser.add_argument("--subset_fraction", type=float, default=1.0,
                        help="Fraction of data to use from each class (e.g., 0.1 = 10%% of data).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    preprocess_dataset(args.data_root, args.split_ratio, args.subset_fraction, args.img_resize, args.seed)

