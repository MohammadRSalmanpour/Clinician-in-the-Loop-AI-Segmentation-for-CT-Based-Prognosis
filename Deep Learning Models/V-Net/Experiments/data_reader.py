
import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class CTScanDataset(Dataset):
    def __init__(self, ct_dir, seg_dir, transform=None):
        """
        Args:
            ct_dir (string): Directory with all the CT images.
            seg_dir (string): Directory with all the segmentation images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.ct_dir = ct_dir
        self.seg_dir = seg_dir
        self.ct_files = sorted(os.listdir(ct_dir))
        self.seg_files = sorted(os.listdir(seg_dir))
        self.transform = transform

        # Ensure that the filenames match in both directories
        assert self.ct_files == self.seg_files, "CT and segmentation filenames do not match."

    def __len__(self):
        return len(self.ct_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ct_filename = self.ct_files[idx]
        seg_filename = self.seg_files[idx]


        ct_path = os.path.join(self.ct_dir, ct_filename)
        seg_path = os.path.join(self.seg_dir, seg_filename)

        ct_image = nib.load(ct_path).get_fdata()
        seg_image = nib.load(seg_path).get_fdata()

        # Add channel dimension
        ct_image = np.expand_dims(ct_image, axis=0)
        seg_image = np.expand_dims(seg_image, axis=0)
        
    
        # Normalize CT images
        # Clip and normalize CT image
        ct_image = self.clip_and_normalize_image(ct_image)
        
        
        # Ensure segmentation mask is binary
        seg_image = np.clip(seg_image, 0, 1)
        
         # Debugging print statements
        #print(f"After normalization CT Image: min={ct_image.min()}, max={ct_image.max()}")
        #print(f"Seg Image: min={seg_image.min()}, max={seg_image.max()}")

        sample = {'ct': ct_image, 'seg': seg_image}

        if self.transform:
            sample = self.transform(sample)

        return sample


    def clip_and_normalize_image(self, image, clip_min=-1000, clip_max=1000):
        # Clip the image to reduce the effect of outliers
        #image = np.clip(image, clip_min, clip_max)
        # Assuming the image is a numpy array of shape (C, D, H, W)
        mean = np.mean(image)
        std = np.std(image)
        normalized_image = (image - mean) / std
        # Debugging print statements
        #print(f"Before normalization - mean={mean}, std={std}")
        return normalized_image
      
        
        
# Example transforms
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        ct, seg = torch.from_numpy(sample['ct']), torch.from_numpy(sample['seg'])
        ct, seg = ct.type(torch.float32), seg.type(torch.float32)
        #print(ct.shape)
        
        x, y, z = 64,64,64 #int((ct.shape[1] * ct.shape[2] * ct.shape[3]) / (256 * 256)) -1  # Example target shape (D, H, W)
        
        target_shape = (1, x, y, z)
        #print("Target shape:", target_shape)

        ct_resized = F.interpolate(ct.unsqueeze(0), size=(x, y, z), mode='trilinear', align_corners=False).squeeze(0)
        seg_resized = F.interpolate(seg.unsqueeze(0), size=(x, y, z), mode='trilinear', align_corners=False).squeeze(0)
        #print('CT: ', ct_resized.dtype)
        #print('Seg: ', seg_resized.dtype)
        
        # Debugging print statements
       # print(f"CT Resized:  shape={ct_resized.shape}")
       # print(f"Seg Resized: shape={seg_resized.shape}")

       

        return {'ct': ct_resized, 'seg': seg_resized}




