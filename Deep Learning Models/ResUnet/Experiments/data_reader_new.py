import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class CTScanDataset(Dataset):
    def __init__(self, ct_dir, seg_dir=None, transform=None):
        """
        Args:
            ct_dir (string): Directory with all the CT images.
            seg_dir (string, optional): Directory with all the segmentation images. If None, only CT images will be loaded.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.ct_dir = ct_dir
        self.seg_dir = seg_dir
        self.ct_files = sorted(os.listdir(ct_dir))
        self.transform = transform

        if seg_dir:
            self.seg_files = sorted(os.listdir(seg_dir))
            # Ensure that the filenames match in both directories
            assert self.ct_files == self.seg_files, "CT and segmentation filenames do not match."
        else:
            self.seg_files = None

    def __len__(self):
        return len(self.ct_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ct_filename = self.ct_files[idx]
        ct_path = os.path.join(self.ct_dir, ct_filename)
        ct_image = nib.load(ct_path).get_fdata()

        # Add channel dimension to the CT image (shape: 1, D, H, W)
        ct_image = np.expand_dims(ct_image, axis=0)

        # Normalize CT image
        ct_image = self.clip_and_normalize_image(ct_image)

        sample = {'ct': ct_image}

        if self.seg_files:
            seg_filename = self.seg_files[idx]
            seg_path = os.path.join(self.seg_dir, seg_filename)
            seg_image = nib.load(seg_path).get_fdata()

            # Add channel dimension to the segmentation image
            seg_image = np.expand_dims(seg_image, axis=0)

            # Ensure segmentation mask is binary
            seg_image = np.clip(seg_image, 0, 1)

            sample['seg'] = seg_image

        # Apply optional transformations
        if self.transform:
            sample = self.transform(sample)

        return sample

    def clip_and_normalize_image(self, image, clip_min=-1000, clip_max=1000):
        """
        Clip the CT image to the specified window and normalize.
        """
        # Clip the image to the desired range to reduce outliers
        #image = np.clip(image, clip_min, clip_max)

        # Normalize the image (zero mean, unit variance)
        mean = np.mean(image)
        std = np.std(image)
        normalized_image = (image - mean) / std

        return normalized_image

# Example transforms
#class ToTensor(object):
   # def __call__(self, sample):
       # """
       # Convert numpy arrays in the sample to PyTorch tensors.
       # """
       # ct, seg = torch.from_numpy(sample['ct']), torch.from_numpy(sample['seg'])
       # ct, seg = ct.type(torch.float32), seg.type(torch.float32)
        #return {'ct': ct, 'seg': seg}
class ToTensor(object):
    def __call__(self, sample):
        """
        Convert numpy arrays in the sample to PyTorch tensors.
        """
        ct = torch.from_numpy(sample['ct']).type(torch.float32)
        sample_out = {'ct': ct}
        
        if 'seg' in sample:
            seg = torch.from_numpy(sample['seg']).type(torch.float32)
            sample_out['seg'] = seg
        
        return sample_out
   

class Resize(object):
    def __init__(self, target_size):
        """
        Initialize the resize transform with the target size.
        """
        self.target_size = target_size

    def __call__(self, sample):
        """
        Resize the CT and segmentation images to the target size.
        """
        ct = sample['ct']

        # Convert NumPy arrays to PyTorch tensors
        ct = torch.from_numpy(ct).unsqueeze(0)

        # Resize the CT image to the target size
        ct = F.interpolate(ct, size=self.target_size, mode='trilinear', align_corners=False).squeeze(0)

        # Update the sample dictionary with the resized CT image
        sample_out = {'ct': ct.numpy()}

        # Check if 'seg' exists in the sample and resize it if present
        if 'seg' in sample:
            seg = sample['seg']
            seg = torch.from_numpy(seg).unsqueeze(0)
            seg = F.interpolate(seg, size=self.target_size, mode='nearest').squeeze(0)
            sample_out['seg'] = seg.numpy()

        return sample_out
