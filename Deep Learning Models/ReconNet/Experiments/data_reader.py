# -*- coding: utf-8 -*-
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
import os

class NiftiDataset(Dataset):
    def __init__(self, image_file_paths, seg_file_paths, transform=None):
        self.image_file_paths = image_file_paths
        self.seg_file_paths = seg_file_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
    
    
        image_path = self.image_file_paths[idx]
    
        seg_path = self.seg_file_paths[idx]
        
        names=os.listdir(image_path)
        
        for name in names:
           image = nib.load(os.path.join(image_path,name)).get_fdata()
           segmentation = nib.load(os.path.join(seg_path,name)).get_fdata()
           
        print('111111111111111111111111111: ', self.file_paths[idx])
        #try:
            #nifti_file = nib.load(self.file_paths[idx])
            #image = nifti_file.get_fdata()
            
            
            
            
           # print('222222222222222222222222222222222222222222222222222222222222: ', self.file_paths[idx])
           # 
       # except:
            #print('3333333333333333333333333333333333333333333333333333333333333333: ', self.file_paths[idx])

        if self.transform:
            image = self.transform(image)
            segmentation = self.transform(segmentation)

        return {
	'image': torch.tensor(image, dtype=torch.float32),
	'segmentation': torch.tensor(segmentation, dtype=torch.float32)
	}


