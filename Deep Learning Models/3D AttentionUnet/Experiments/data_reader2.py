# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 00:28:49 2024

@author: Amin
"""
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NiftiDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        nifti_file = nib.load(self.file_paths[idx])
        image = nifti_file.get_fdata()

        
        # Print the size of the image before cropping
        print(f"Original image size: {image.shape}")

        # Perform center cropping
        image = self.center_crop(image.unsqueeze(0)).squeeze(0)

        if self.transform:
            image = self.transform(image)


        return torch.tensor(image, dtype=torch.float32)


