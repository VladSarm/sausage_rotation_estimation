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
from angle_estimation_model import mean_shift

def vec2deg(v: torch.Tensor) -> torch.Tensor:
    """v is (N,2) on CPU; returns angles in [0,360)"""
    ang = torch.atan2(v[:,1], v[:,0])   # (-π, π]
    deg = torch.rad2deg(ang) % 360
    return deg

def angle_to_unitvec(label: int) -> torch.Tensor:
    """label is an int 0…359 coming from ImageFolder"""
    rad = math.radians(label)
    return torch.tensor([math.cos(rad), math.sin(rad)], dtype=torch.float32)

def angle_to_vec(angle_deg: torch.Tensor) -> torch.Tensor:
    rad = torch.deg2rad(angle_deg)
    return torch.stack([torch.cos(rad), torch.sin(rad)], dim=1)  # [B,2]

class EnsembleDataset(Dataset):
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
        
        return image, torch.tensor(angle, dtype=torch.float32)

class AngleEnsemble(nn.Module):
    """
    model_a и model_b – ваши две готовые сети, каждая возвращает
    тензор формы [B, 2] с нормированным (cos, sin).
    train_backbone=False замораживает их параметры.
    """
    def __init__(self, model_von: nn.Module, model_sin: nn.Module,
                 train_backbone: bool = False):
        super().__init__()
        self.model_von = model_von
        self.model_sin = model_sin

        if not train_backbone:
            for p in self.model_von.parameters():
                p.requires_grad = False
            for p in self.model_sin.parameters():
                p.requires_grad = False

        # 4 → 128 → 2, можно расширить/сузить по желанию
        self.head = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        with torch.no_grad():             # бэкбоны заморожены
            sin_vec = self.model_sin(x)    # [B,2]  (cos,sin)

            von_logits = self.model_von(x) # list[M] of [B,N]
            batch_angles = []
            for i in range(x.size(0)):
                sample = [F.softmax(von_logits[m][i], dim=0) for m in range(self.model_von.M)]
                ang_i = mean_shift(sample, self.model_von.N, self.model_von.M)  # float
                batch_angles.append(ang_i)
            von_vec = angle_to_vec(torch.tensor(batch_angles, device=x.device))  # [B,2]

        fused = torch.cat([sin_vec, von_vec], dim=1)  # [B,4]
        out   = self.head(fused)                      # [B,2]
        return F.normalize(out, p=2, dim=1)