import torch

class SimpleCNNDataset(torch.utils.data.Dataset):
    def __init__(self, folder_dataset, transform=None):
        self.folder_dataset = folder_dataset
        self.transform = transform
        self.classes = folder_dataset.classes
        self.class_to_idx = folder_dataset.class_to_idx

    def __len__(self):
        return len(self.folder_dataset)

    def __getitem__(self, idx):
        img, label = self.folder_dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label  # plain tuple, compatible with manual training loop

def load_datasets():
    import yaml
    from torchvision import datasets, transforms
    import torch

    with open("./configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]

    train_transforms = transforms.Compose([
        transforms.Resize((data_cfg["img_size"], data_cfg["img_size"])),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(25),
        # transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((data_cfg["img_size"], data_cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_path = f"{data_cfg['root']}/train"
    val_path   = f"{data_cfg['root']}/val"
    test_path  = f"{data_cfg['root']}/test"

    train_dataset = SimpleCNNDataset(datasets.ImageFolder(root=train_path), transform=train_transforms)
    val_dataset   = SimpleCNNDataset(datasets.ImageFolder(root=val_path), transform=val_transforms)
    test_dataset  = SimpleCNNDataset(datasets.ImageFolder(root=test_path), transform=val_transforms)

    labels = train_dataset.classes

    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    print(f"✅ Data loaded from: {data_cfg['root']}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"🧠 Using device: {device}")

    return train_dataset, val_dataset, test_dataset, labels, device
