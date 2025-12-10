import yaml
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import sklearn as sk

import kagglehub

class HFImageFolder(torch.utils.data.Dataset):
    def __init__(self, folder_dataset):
        self.folder_dataset = folder_dataset
        self.classes = folder_dataset.classes
        self.class_to_idx = folder_dataset.class_to_idx

    def __len__(self):
        return len(self.folder_dataset)

    def __getitem__(self, idx):
        img, label = self.folder_dataset[idx]
        return {
            "pixel_values": img,
            "labels": label
        }
    
def split_data_into_fold(dataset, k: int, seed: int):
    """
    Split dataset randomly into k folds to perform k-fold cross validation. 
    Random split through a given seed.
    """
    train_dataset = None
    val_dataset = None

    return train_dataset, val_dataset
    
def load_datasets():
    # --- Load base.yaml ---
    with open("./configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]

    # --- Define transforms (should match your preprocessing normalization) ---
    transform = transforms.Compose([
        transforms.Resize((data_cfg["img_size"], data_cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- Dataset paths from config ---
    train_path = f"{data_cfg['root']}/train"
    val_path   = f"{data_cfg['root']}/val"
    test_path  = f"{data_cfg['root']}/test"

    # --- Load datasets ---
    train_dataset = HFImageFolder(datasets.ImageFolder(root=train_path, transform=transform))
    val_dataset   = HFImageFolder(datasets.ImageFolder(root=val_path, transform=transform))
    test_dataset  = HFImageFolder(datasets.ImageFolder(root=test_path, transform=transform))

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=data_cfg["batch_size"], shuffle=True, num_workers=data_cfg["num_workers"])
    val_loader   = DataLoader(val_dataset, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"])
    test_loader  = DataLoader(test_dataset, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"])

    labels = train_dataset.classes
    # --- Example usage ---
    print(f"✅ Data loaded from: {data_cfg['root']}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # --- Device setup (optional, from config) ---
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() or config["training"]["device"] == "mps" else "cpu")
    print(f"🧠 Using device: {device}")

    return train_dataset, val_dataset, test_dataset, labels


if __name__ == "__main__":
    # load datasets
    # Download latest version
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")

    print("Path to dataset files:", path)
