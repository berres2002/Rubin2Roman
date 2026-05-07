import os
import pandas as pd
import numpy as np
from torchvision.io import decode_image
from torch.utils.data import Dataset
import torch

def _ZScoreNormalize(image: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each channel of the input image independently.
    
    Parameters:
    image (np.ndarray): Input image of shape (C, H, W) where C is the number of channels.
    
    Returns:
    np.ndarray: Z-score normalized image of the same shape as the input.
    """
    mean = np.mean(image, axis=(1, 2), keepdims=True)
    std = np.std(image, axis=(1, 2), keepdims=True)
    return (image - mean) / std
    

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) # This file contain image filenames
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_labels.iloc[idx,0]) # Point this to individual image paths
        # image = decode_image(img_path)
        image = torch.from_numpy(np.load(img_path))
        image=image.unsqueeze(0)  # Add channel dimension if needed
        image = image.to(device='cuda')  # Load .npy file as tensor
        # label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image
    
class CustomImageDatasetCond(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) # This file contain image filenames
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_labels.iloc[idx,0]) # Point this to individual image paths
        # image = decode_image(img_path)
        image_load = np.load(img_path)
        roman_channels = image_load[6:] # Roman channels as the target image
        rubin_channels = image_load[:6] # Rubin channels as conditioning information
        roman_norm = _ZScoreNormalize(roman_channels) # shape (9, 64, 64) for 9 channels (6 Rubin + 3 Roman) and 64x64 cutout size
        rubin_norm = _ZScoreNormalize(rubin_channels)
        # image_full = torch.from_numpy() # shape (9, 64, 64) for 9 channels (6 Rubin + 3 Roman) and 64x64 cutout size
        image_cond = torch.from_numpy(rubin_norm)#image_full[:6] # Rubin channels as conditioning information
        image = torch.from_numpy(roman_norm)#image_full[6:] # Roman channels as
        # image=image.unsqueeze(0)  # Add channel dimension if needed
        image = image.to(device='cuda')  # Load .npy file as tensor
        image_cond = image_cond.to(device='cuda')  # Load .npy file as tensor
        # label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, image_cond
    
class CustomImageDatasetYJH(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) # This file contain image filenames
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_labels.iloc[idx,0]) # Point this to individual image paths
        # image = decode_image(img_path)
        image_full = torch.from_numpy(np.load(img_path)) # shape (9, 64, 64) for 9 channels (6 Rubin + 3 Roman) and 64x64 cutout size
        # image_cond = image_full[:6] # Rubin channels as conditioning information
        image = image_full[6:] # Roman channels as
        # image=image.unsqueeze(0)  # Add channel dimension if needed
        image = image.to(device='cuda')  # Load .npy file as tensor
        # image_cond = image_cond.to(device='cuda')  # Load .npy file as tensor
        # label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image
