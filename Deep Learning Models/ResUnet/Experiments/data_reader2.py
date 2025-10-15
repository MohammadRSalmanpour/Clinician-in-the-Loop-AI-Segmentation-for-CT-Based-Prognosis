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
        
        
        
        
 this is datareader==# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 01:19:00 2024

@author: Amin
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 00:28:49 2024


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
        print('111111111111111111111111111: ', self.file_paths[idx])
        try:
            nifti_file = nib.load(self.file_paths[idx])
            image = nifti_file.get_fdata()
            print('222222222222222222222222222222222222222222222222222222222222: ', self.file_paths[idx])
            
        except:
            print('3333333333333333333333333333333333333333333333333333333333333333: ', self.file_paths[idx])

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32)       
        
        
        
        


