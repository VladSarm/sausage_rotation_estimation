import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.special import i0  # Modified Bessel function of order 0

def angle_to_unitvec(label: int) -> torch.Tensor:
    """label is an int 0…359 coming from ImageFolder"""
    rad = math.radians(label)
    return torch.tensor([math.cos(rad), math.sin(rad)], dtype=torch.float32)

class SinCosDataset(Dataset):
    """
    Dataset for loading images with angle labels from folders named 0-359
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            discretization_params (dict): Parameters for angle discretization (N, M)
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.samples = []
        
        # Find all images and their respective angles
        sorted_folders = sorted(os.listdir(root_dir), key=lambda x: int(x) if x.isdigit() else -1)
        for angle_folder in sorted_folders:
            try:
                angle = int(angle_folder)
                if 0 <= angle <= 359:
                    folder_path = os.path.join(root_dir, angle_folder)
                    if os.path.isdir(folder_path):
                        for img_name in os.listdir(folder_path):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.samples.append((os.path.join(folder_path, img_name), angle))
            except ValueError:
                # Not a number, skip this folder
                continue
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, angle = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, angle_to_unitvec(angle)

class SinCosModel(nn.Module):
    """
    CNN для изображений 224×224, выдаёт (cos, sin), нормализованные к длине 1.
    """
    def __init__(self, input_channels: int = 3):
        super().__init__()

        # ── ❶ Экстрактор признаков ──────────────────────────────────────────────
        self.features = nn.Sequential(
            # Block 1: 224×224 → 112×112
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 112×112 → 56×56
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 56×56 → 28×28
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 28×28 → 14×14
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Глобальный pooling: 14×14 → 1×1
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ── ❷ Классификатор ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),          # 128 × 1 × 1 → 128
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)      # выход: (cos, sin)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (B,128,14,14)
        x = self.global_pool(x)    # (B,128,1,1)
        x = self.classifier(x)     # (B,2)
        return F.normalize(x, p=2, dim=1)
    
class SinCosResModel(nn.Module):
    def __init__(self, feature_extract: bool = True):
        super().__init__()
        backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        if feature_extract:
            for p in backbone.parameters():
                p.requires_grad = False

            for p in backbone.layer4.parameters():
                p.requires_grad = True

        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)