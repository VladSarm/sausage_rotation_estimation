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

class AngleDataset(Dataset):
    """
    Dataset for loading images with angle labels from folders named 0-359
    """
    def __init__(self, root_dir, transform=None, discretization_params=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            discretization_params (dict): Parameters for angle discretization (N, M)
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Generate discretization parameters
        if discretization_params is None:
            self.N = 8  # Number of orientations per task
            self.M = 9  # Number of tasks (different starting angles)
        else:
            self.N = discretization_params['N']
            self.M = discretization_params['M']
        
        # Calculate granularity G
        self.G = 360 / self.N
        
        self.samples = []
        
        # Find all images and their respective angles
        for angle_folder in os.listdir(root_dir):
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
        
        # Convert the continuous angle to M different N-way classification tasks
        # For each task m, find the closest discrete angle
        target_classes = []
        for m in range(self.M):
            # Starting angle for task m
            start_angle = m * self.G / self.M
            
            # Find the closest discrete angle in task m to the true angle
            discrete_angles = [(start_angle + k * self.G) % 360 for k in range(self.N)]
            
            # Find the closest discrete angle
            class_idx = np.argmin([(angle - a + 180) % 360 - 180 for a in discrete_angles])
            target_classes.append(class_idx)
        
        return image, torch.tensor(target_classes), torch.tensor(angle, dtype=torch.float32)
    
class OrientationPredictionUnit(nn.Module):
    """
    Implementation of Approach 3 from the paper
    """
    def __init__(self, input_dim, N=8, M=9):
        """
        Args:
            input_dim: Dimension of the input feature map
            N: Number of discrete orientations per task
            M: Number of tasks (different starting orientations)
        """
        super(OrientationPredictionUnit, self).__init__()
        self.N = N
        self.M = M
        
        # M independent N-way classification layers
        self.classifiers = nn.ModuleList([
            nn.Linear(input_dim, N) for _ in range(M)
        ])
    
    def forward(self, x):
        # Apply each classifier
        outputs = [classifier(x) for classifier in self.classifiers]
        return outputs  # List of M tensors, each of shape [batch_size, N]

class AngleEstimationModel(nn.Module):
    def __init__(self, N=8, M=9, feature_extract=True):
        """
        Args:
            N: Number of discrete orientations
            M: Number of classification tasks
            feature_extract: If True, only update the reshaped layer params, otherwise fine-tune the whole model
        """
        super(AngleEstimationModel, self).__init__()
        self.N = N
        self.M = M
        
        # Load the pretrained model
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        
        # Set gradients to be frozen for all parameters unless feature_extract is False
        if feature_extract:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get the number of features in the final layer
        num_ftrs = self.model.fc.in_features
        
        # Replace the final fully connected layer
        self.model.fc = nn.Identity()
        
        # Average pooling with size 3 and stride 1
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        # Orientation prediction unit
        self.orientation_unit = OrientationPredictionUnit(num_ftrs, N, M)
        
    def forward(self, x):
        # Get features from the ResNet backbone (up to the layer before fc)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        # Apply average pooling
        x = self.avg_pool(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply orientation prediction unit
        return self.orientation_unit(x)

def von_mises_kernel(theta, nu=1):
    """
    Von Mises kernel as defined in the paper
    Args:
        theta: Angle in radians
        nu: Concentration parameter
    """
    return 1/(2*math.pi*i0(nu)) * np.exp(nu * np.cos(theta))

def mean_shift(softmax_outputs, N, M, nu=1):
    """
    Mean-shift algorithm to find the continuous angle estimation
    Args:
        softmax_outputs: List of M softmax probability arrays, each of shape [N]
        N: Number of discrete orientations per task
        M: Number of tasks
        nu: Concentration parameter for the von Mises kernel
    Returns:
        Estimated continuous angle in degrees
    """
    # Initialize at the angle with highest probability
    all_probs = []
    all_angles = []
    
    G = 360 / N
    
    # Collect all discrete angles and their probabilities
    for m in range(M):
        start_angle = m * G / M
        for k in range(N):
            angle_degrees = (start_angle + k * G) % 360
            angle_radians = math.radians(angle_degrees)
            all_angles.append(angle_radians)
            all_probs.append(softmax_outputs[m][k].item())
    
    # Find the angle with the highest probability
    max_prob_idx = np.argmax(all_probs)
    theta = all_angles[max_prob_idx]
    
    # Mean-shift iterations
    max_iterations = 100
    epsilon = 0.01  # convergence threshold in radians
    
    for _ in range(max_iterations):
        # Calculate the density using von Mises kernel
        kernel_values = []
        for angle in all_angles:
            # Distance in circular space
            dist = abs(theta - angle)
            dist = min(dist, 2*math.pi - dist)
            kernel_values.append(von_mises_kernel(dist, nu))
        
        # Calculate the mean shift
        numerator = 0
        denominator = 0
        
        for i, angle in enumerate(all_angles):
            w = all_probs[i] * kernel_values[i]
            numerator += w * angle
            denominator += w
        
        if denominator > 0:
            new_theta = numerator / denominator
        else:
            break
        
        # Check for convergence
        if abs(new_theta - theta) < epsilon:
            theta = new_theta
            break
        
        theta = new_theta
    
    # Convert back to degrees
    angle_degrees = math.degrees(theta) % 360

    # MY FIX. Somehow error was bigger on 180 degrees.
    angle_degrees += 180
    if angle_degrees > 360:
        angle_degrees -= 360

    return angle_degrees

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Function to train the model
    """
    model.train()
    
    total_mae = 0.0
    count = 0
    running_loss = 0.0
    
    for inputs, targets, true_angles in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        true_angles = true_angles.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss (sum of losses from M classifiers)
        loss = 0
        for m in range(model.M):
            loss += criterion(outputs[m], targets[:, m])
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)

        # Convert softmax outputs to angles
        batch_size = inputs.size(0)
        for i in range(batch_size):
            # Extract predictions for this sample
            sample_outputs = []
            for m in range(model.M):
                softmax_probs = F.softmax(outputs[m][i], dim=0)
                sample_outputs.append(softmax_probs)
            
            # Apply mean-shift to get continuous angle
            predicted_angle = mean_shift(sample_outputs, model.N, model.M)
            
            # Calculate angular error (minimum difference in circle)
            true_angle = true_angles[i].item()
            error = abs((predicted_angle - true_angle + 180) % 360 - 180)
            
            total_mae += error
            count += 1
    
    mae = total_mae / count
    
    return epoch_loss, mae

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model and calculate the Mean Absolute Error (MAE)
    """
    model.eval()
    
    total_mae = 0.0
    count = 0
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets, true_angles in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            true_angles = true_angles.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = 0
            for m in range(model.M):
                loss += criterion(outputs[m], targets[:, m])
            
            running_loss += loss.item()
            epoch_loss = running_loss / len(dataloader)
            
            # Convert softmax outputs to angles
            batch_size = inputs.size(0)
            for i in range(batch_size):
                # Extract predictions for this sample
                sample_outputs = []
                for m in range(model.M):
                    softmax_probs = F.softmax(outputs[m][i], dim=0)
                    sample_outputs.append(softmax_probs)
                
                # Apply mean-shift to get continuous angle
                predicted_angle = mean_shift(sample_outputs, model.N, model.M)
                
                # Calculate angular error (minimum difference in circle)
                true_angle = true_angles[i].item()
                error = abs((predicted_angle - true_angle + 180) % 360 - 180)
                
                total_mae += error
                count += 1
    
    mae = total_mae / count
    return epoch_loss, mae

def split_dataset(source_folder, dest_base_folder, val_ratio=0.1, random_seed=42):
    """
    Split a dataset of images organized in angle folders (0-359) into train and validation sets.
    Creates or clears train and val folders in the destination directory.
    
    Args:
        source_folder (str): Path to the source folder containing angle folders (0-359)
        dest_base_folder (str): Path to destination base folder where train and val folders will be created
        val_ratio (float): Ratio of validation samples (0.0 to 1.0)
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_count, val_count) - Number of images in training and validation sets
    """
    import os
    import shutil
    import random
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create or clear train and val folders
    train_folder = os.path.join(dest_base_folder, 'train')
    val_folder = os.path.join(dest_base_folder, 'val')
    
    # Clear existing folders if they exist
    for folder in [train_folder, val_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
    
    # Track counts
    train_count = 0
    val_count = 0
    
    # Process each angle folder
    for angle_folder in os.listdir(source_folder):
        try:
            # Check if the folder name is a valid angle (0-359)
            angle = int(angle_folder)
            if 0 <= angle <= 359:
                src_angle_path = os.path.join(source_folder, angle_folder)
                if os.path.isdir(src_angle_path):
                    # Create the corresponding angle folders in train and val
                    train_angle_path = os.path.join(train_folder, angle_folder)
                    val_angle_path = os.path.join(val_folder, angle_folder)
                    
                    os.makedirs(train_angle_path, exist_ok=True)
                    os.makedirs(val_angle_path, exist_ok=True)
                    
                    # Get all image files in this angle folder
                    image_files = [f for f in os.listdir(src_angle_path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    # Shuffle images
                    random.shuffle(image_files)
                    
                    # Calculate split point
                    val_size = max(1, int(len(image_files) * val_ratio))
                    
                    # Split into train and validation sets
                    val_files = image_files[:val_size]
                    train_files = image_files[val_size:]
                    
                    # Copy files to respective folders
                    for f in train_files:
                        shutil.copy2(os.path.join(src_angle_path, f), 
                                    os.path.join(train_angle_path, f))
                        train_count += 1
                    
                    for f in val_files:
                        shutil.copy2(os.path.join(src_angle_path, f), 
                                    os.path.join(val_angle_path, f))
                        val_count += 1
                    
        except ValueError:
            # Not a valid angle folder, skip
            continue
    
    print(f"Dataset split complete: {train_count} training images, {val_count} validation images")
    return train_count, val_count
