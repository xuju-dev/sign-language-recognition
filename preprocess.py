import os
import random
import shutil
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

def preprocess_dataset(root, split_ratio=0.8):
    splits = ['train', 'test']

    # Create preprocessed directories
    for split in splits:
        preprocessed_dir = os.path.join(root, f'asl_alphabet_preprocessed')
        split_dir = os.path.join(preprocessed_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        if split == 'train':
            # Paths
            train_dir = os.path.join(root, 'train')
            train_pre = os.path.join(preprocessed_dir, 'train')
            val_pre = os.path.join(preprocessed_dir, 'val')

            # Create val directory
            os.makedirs(val_pre, exist_ok=True)

            total_images = sum(
                len(files) for _, _, files in os.walk(train_dir)
            )
            # Get class subfolders
            for cls in os.listdir(train_dir):
                cls_path = os.path.join(train_dir, cls)
                if not os.path.isdir(cls_path):
                    continue

            # Create class subfolders in new dirs
            os.makedirs(os.path.join(train_pre, cls), exist_ok=True)
            os.makedirs(os.path.join(val_pre, cls), exist_ok=True)

            # List images and split according to ratio
            imgs = os.listdir(cls_path)
            random.shuffle(imgs)
            split_idx = int(len(imgs) * split_ratio)
            train_imgs = imgs[:split_idx]
            val_imgs = imgs[split_idx:]

            # Preprocess train and val images
            # TODO: adjust progress bar in train folder
            for subset, img_list in tqdm([('train', train_imgs), ('val', val_imgs)], desc="Preprocessing train and val folder", ncols=100):
                for img_name in img_list:
                    src = os.path.join(cls_path, img_name)
                    dst = os.path.join(preprocessed_dir, subset, cls, img_name)

                    # Resize and normalize
                    img = Image.open(src).convert('RGB')
                    img = img.resize((128, 128))
                    arr = np.asarray(img) / 255.0  # normalize to [0, 1]
                    img = Image.fromarray((arr * 255).astype(np.uint8))
                    img.save(dst)

        if split == 'test':
            test_dir = os.path.join(root, 'test')
            test_pre = os.path.join(preprocessed_dir, 'test')

            total_images = sum(
                len(files) for _, _, files in os.walk(test_dir)
            )

            with tqdm(total=total_images, desc="Preprocessing test folder", ncols=100) as pbar:

                # for cls in os.listdir(test_dir):
                #     cls_path = os.path.join(test_dir, cls)
                #     if not os.path.isdir(cls_path):
                #         continue
                #     os.makedirs(os.path.join(test_pre, cls), exist_ok=True)

                for img_name in os.listdir(test_dir):
                    # src = os.path.join(cls_path, img_name)
                    src = os.path.join(test_dir, img_name)
                    dst = os.path.join(test_pre, img_name)

                    # Resize and normalize
                    img = Image.open(src).convert('RGB')
                    img = img.resize((128, 128))
                    arr = np.asarray(img) / 255.0
                    img = Image.fromarray((arr * 255).astype(np.uint8))
                    img.save(dst)
                    pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset and split train/val sets.")
    parser.add_argument("--data_root", type=str, default='data/asl_alphabet_dataset', help="Root dataset directory.")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Train/validation split ratio (default: 0.8 for 80/20).")

    args = parser.parse_args()
    preprocess_dataset(args.data_root, args.split_ratio)
