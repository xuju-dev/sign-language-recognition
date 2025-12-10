import torch
import torch.nn as nn
import requests
from PIL import Image
import timm

class HFImageClassifier(nn.Module):
    """
    Wrap a timm model so it works with HuggingFace Trainer.
    Expects dataset items with keys 'pixel_values' and 'labels'.
    """
    def __init__(self, backbone: nn.Module, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_labels = num_labels

    def forward(self, pixel_values=None, labels=None, **kwargs):
        # timm models take (B, C, H, W)
        logits = self.backbone(pixel_values)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

def load_mobilev3_model(activation_variant: str, num_labels: int, device):
    """
    Load a MobileNetV3 model variant with a specified number of output labels.
    
    Args:
        activation_variant (str): The activation function variant of MobileNetV3 to load. Options are 'orignial', 'relu', 'leakyrelu'.
        
    Returns:
        model (AutoModelForImageClassification): The loaded MobileNetV3 model.
    """
    activation_name_map = {
        'original': nn.Hardswish(),
        'relu': nn.ReLU(),
        'leakyrelu': nn.LeakyReLU(),
    }
    
    if activation_variant not in activation_name_map:
        raise ValueError(f"Unsupported activation variant '{activation_variant}'. Choose from {list(activation_name_map.keys())}.")
    
    activation = activation_name_map[activation_variant]

    backbone = timm.create_model(
        "mobilenetv3_small_100.lamb_in1k",
        pretrained=True,
        num_classes=num_labels,
    )
    backbone.act2 = activation_name_map[activation_variant]
    backbone.to(device)

    model = HFImageClassifier(backbone=backbone, num_labels=num_labels)
    
    print(f"Loaded MobileNetV3 model with '{activation_variant}' activation in classifier head.")
    # print(model)
    return model

if __name__ == "__main__":
    # Example usage
    model = load_mobilev3_model('leakyrelu', num_labels=10, device='cpu')
